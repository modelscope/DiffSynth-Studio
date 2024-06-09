from ..models.hunyuan_dit import HunyuanDiT
from ..models.hunyuan_dit_text_encoder import HunyuanDiTCLIPTextEncoder, HunyuanDiTT5TextEncoder
from ..models.sdxl_vae_encoder import SDXLVAEEncoder
from ..models.sdxl_vae_decoder import SDXLVAEDecoder
from ..models import ModelManager
from ..prompts import HunyuanDiTPrompter
from ..schedulers import EnhancedDDIMScheduler
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np



class ImageSizeManager:
    def __init__(self):
        pass


    def _to_tuple(self, x):
        if isinstance(x, int):
            return x, x
        else:
            return x


    def get_fill_resize_and_crop(self, src, tgt):
        th, tw = self._to_tuple(tgt)
        h, w = self._to_tuple(src)

        tr = th / tw        # base 分辨率
        r = h / w           # 目标分辨率

        # resize
        if r > tr:
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))    # 根据base分辨率，将目标分辨率resize下来

        crop_top = int(round((th - resize_height) / 2.0))
        crop_left = int(round((tw - resize_width) / 2.0))

        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


    def get_meshgrid(self, start, *args):
        if len(args) == 0:
            # start is grid_size
            num = self._to_tuple(start)
            start = (0, 0)
            stop = num
        elif len(args) == 1:
            # start is start, args[0] is stop, step is 1
            start = self._to_tuple(start)
            stop = self._to_tuple(args[0])
            num = (stop[0] - start[0], stop[1] - start[1])
        elif len(args) == 2:
            # start is start, args[0] is stop, args[1] is num
            start = self._to_tuple(start)       # 左上角   eg: 12,0
            stop = self._to_tuple(args[0])      # 右下角   eg: 20,32
            num = self._to_tuple(args[1])       # 目标大小  eg: 32,124
        else:
            raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

        grid_h = np.linspace(start[0], stop[0], num[0], endpoint=False, dtype=np.float32) # 12-20 中间差值32份   0-32 中间差值124份
        grid_w = np.linspace(start[1], stop[1], num[1], endpoint=False, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)   # [2, W, H]
        return grid


    def get_2d_rotary_pos_embed(self, embed_dim, start, *args, use_real=True):
        grid = self.get_meshgrid(start, *args)   # [2, H, w]
        grid = grid.reshape([2, 1, *grid.shape[1:]])   # 返回一个采样矩阵  分辨率与目标分辨率一致
        pos_embed = self.get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
        return pos_embed


    def get_2d_rotary_pos_embed_from_grid(self, embed_dim, grid, use_real=False):
        assert embed_dim % 4 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_rotary_pos_embed(embed_dim // 2, grid[0].reshape(-1), use_real=use_real)  # (H*W, D/4)
        emb_w = self.get_1d_rotary_pos_embed(embed_dim // 2, grid[1].reshape(-1), use_real=use_real)  # (H*W, D/4)

        if use_real:
            cos = torch.cat([emb_h[0], emb_w[0]], dim=1)    # (H*W, D/2)
            sin = torch.cat([emb_h[1], emb_w[1]], dim=1)    # (H*W, D/2)
            return cos, sin
        else:
            emb = torch.cat([emb_h, emb_w], dim=1)    # (H*W, D/2)
            return emb


    def get_1d_rotary_pos_embed(self, dim: int, pos, theta: float = 10000.0, use_real=False):
        if isinstance(pos, int):
            pos = np.arange(pos)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [D/2]
        t = torch.from_numpy(pos).to(freqs.device)  # type: ignore  # [S]
        freqs = torch.outer(t, freqs).float()  # type: ignore   # [S, D/2]
        if use_real:
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
            return freqs_cos, freqs_sin
        else:
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
            return freqs_cis
        

    def calc_rope(self, height, width):
        patch_size = 2
        head_size = 88
        th = height // 8 // patch_size
        tw = width // 8 // patch_size
        base_size = 512 // 8 // patch_size
        start, stop = self.get_fill_resize_and_crop((th, tw), base_size)
        sub_args = [start, stop, (th, tw)]
        rope = self.get_2d_rotary_pos_embed(head_size, *sub_args)
        return rope



class HunyuanDiTImagePipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.scheduler = EnhancedDDIMScheduler(prediction_type="v_prediction", beta_start=0.00085, beta_end=0.03)
        self.prompter = HunyuanDiTPrompter()
        self.device = device
        self.torch_dtype = torch_dtype
        self.image_size_manager = ImageSizeManager()
        # models
        self.text_encoder: HunyuanDiTCLIPTextEncoder = None
        self.text_encoder_t5: HunyuanDiTT5TextEncoder = None
        self.dit: HunyuanDiT = None
        self.vae_decoder: SDXLVAEDecoder = None
        self.vae_encoder: SDXLVAEEncoder = None


    def fetch_main_models(self, model_manager: ModelManager):
        self.text_encoder = model_manager.hunyuan_dit_clip_text_encoder
        self.text_encoder_t5 = model_manager.hunyuan_dit_t5_text_encoder
        self.dit = model_manager.hunyuan_dit
        self.vae_decoder = model_manager.vae_decoder
        self.vae_encoder = model_manager.vae_encoder


    def fetch_prompter(self, model_manager: ModelManager):
        self.prompter.load_from_model_manager(model_manager)


    @staticmethod
    def from_model_manager(model_manager: ModelManager):
        pipe = HunyuanDiTImagePipeline(
            device=model_manager.device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_main_models(model_manager)
        pipe.fetch_prompter(model_manager)
        return pipe
    

    def preprocess_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
        return image
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        image = image.cpu().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image
    

    def prepare_extra_input(self, height=1024, width=1024, tiled=False, tile_size=64, tile_stride=32, batch_size=1):
        if tiled:
            height, width = tile_size * 16, tile_size * 16
        image_meta_size = torch.as_tensor([width, height, width, height, 0, 0]).to(device=self.device)
        freqs_cis_img = self.image_size_manager.calc_rope(height, width)
        image_meta_size = torch.stack([image_meta_size] * batch_size)
        return {
            "size_emb": image_meta_size,
            "freq_cis_img": (freqs_cis_img[0].to(dtype=self.torch_dtype, device=self.device), freqs_cis_img[1].to(dtype=self.torch_dtype, device=self.device)),
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride
        }
    

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        cfg_scale=7.5,
        clip_skip=1,
        clip_skip_2=1,
        input_image=None,
        reference_images=[],
        reference_strengths=[0.4],
        denoising_strength=1.0,
        height=1024,
        width=1024,
        num_inference_steps=20,
        tiled=False,
        tile_size=64,
        tile_stride=32,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        noise = torch.randn((1, 4, height//8, width//8), device=self.device, dtype=self.torch_dtype)
        if input_image is not None:
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.torch_dtype)
            latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(self.torch_dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise.clone()

        # Prepare reference latents
        reference_latents = []
        for reference_image in reference_images:
            reference_image = self.preprocess_image(reference_image).to(device=self.device, dtype=self.torch_dtype)
            reference_latents.append(self.vae_encoder(reference_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(self.torch_dtype))

        # Encode prompts
        prompt_emb_posi, attention_mask_posi, prompt_emb_t5_posi, attention_mask_t5_posi = self.prompter.encode_prompt(
            self.text_encoder,
            self.text_encoder_t5,
            prompt,
            clip_skip=clip_skip,
            clip_skip_2=clip_skip_2,
            positive=True,
            device=self.device
        )
        if cfg_scale != 1.0:
            prompt_emb_nega, attention_mask_nega, prompt_emb_t5_nega, attention_mask_t5_nega = self.prompter.encode_prompt(
                self.text_encoder,
                self.text_encoder_t5,
                negative_prompt,
                clip_skip=clip_skip,
                clip_skip_2=clip_skip_2,
                positive=False,
                device=self.device
            )

        # Prepare positional id
        extra_input = self.prepare_extra_input(height, width, tiled, tile_size)

        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.tensor([timestep]).to(dtype=self.torch_dtype, device=self.device)

            # In-context reference
            for reference_latents_, reference_strength in zip(reference_latents, reference_strengths):
                if progress_id < num_inference_steps * reference_strength:
                    noisy_reference_latents = self.scheduler.add_noise(reference_latents_, noise, self.scheduler.timesteps[progress_id])
                    self.dit(
                        noisy_reference_latents,
                        prompt_emb_posi, prompt_emb_t5_posi, attention_mask_posi, attention_mask_t5_posi,
                        timestep,
                        **extra_input,
                        to_cache=True
                    )
            # Positive side
            noise_pred_posi = self.dit(
                latents,
                prompt_emb_posi, prompt_emb_t5_posi, attention_mask_posi, attention_mask_t5_posi,
                timestep,
                **extra_input,
            )
            if cfg_scale != 1.0:
                # Negative side
                noise_pred_nega = self.dit(
                    latents,
                    prompt_emb_nega, prompt_emb_t5_nega, attention_mask_nega, attention_mask_t5_nega,
                    timestep,
                    **extra_input
                )
                # Classifier-free guidance
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        image = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        return image
