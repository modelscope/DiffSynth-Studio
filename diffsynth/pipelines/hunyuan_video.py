from ..models import ModelManager, SD3TextEncoder1, HunyuanVideoVAEDecoder, HunyuanVideoVAEEncoder
from ..models.hunyuan_video_dit import HunyuanVideoDiT
from ..models.hunyuan_video_text_encoder import HunyuanVideoLLMEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import HunyuanVideoPrompter
import torch
import torchvision.transforms as transforms
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm


class HunyuanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=7.0, sigma_min=0.0, extra_one_step=True)
        self.prompter = HunyuanVideoPrompter()
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: HunyuanVideoLLMEncoder = None
        self.dit: HunyuanVideoDiT = None
        self.vae_decoder: HunyuanVideoVAEDecoder = None
        self.vae_encoder: HunyuanVideoVAEEncoder = None
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'dit', 'vae_decoder', 'vae_encoder']
        self.vram_management = False


    def enable_vram_management(self):
        self.vram_management = True
        self.enable_cpu_offload()
        self.text_encoder_2.enable_auto_offload(dtype=self.torch_dtype, device=self.device)
        self.dit.enable_auto_offload(dtype=self.torch_dtype, device=self.device)


    def fetch_models(self, model_manager: ModelManager):
        self.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("hunyuan_video_text_encoder_2")
        self.dit = model_manager.fetch_model("hunyuan_video_dit")
        self.vae_decoder = model_manager.fetch_model("hunyuan_video_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("hunyuan_video_vae_encoder")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, enable_vram_management=True):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = HunyuanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if enable_vram_management:
            pipe.enable_vram_management()
        return pipe

    def generate_crop_size_list(self, base_size=256, patch_size=32, max_ratio=4.0):
        num_patches = round((base_size / patch_size)**2)
        assert max_ratio >= 1.0
        crop_size_list = []
        wp, hp = num_patches, 1
        while wp > 0:
            if max(wp, hp) / min(wp, hp) <= max_ratio:
                crop_size_list.append((wp * patch_size, hp * patch_size))
            if (hp + 1) * wp <= num_patches:
                hp += 1
            else:
                wp -= 1
        return crop_size_list


    def get_closest_ratio(self, height: float, width: float, ratios: list, buckets: list):
        aspect_ratio = float(height) / float(width)
        closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
        closest_ratio = min(ratios, key=lambda ratio: abs(float(ratio) - aspect_ratio))
        return buckets[closest_ratio_id], float(closest_ratio)


    def prepare_vae_images_inputs(self, semantic_images, i2v_resolution="720p"):
        if i2v_resolution == "720p":
            bucket_hw_base_size = 960
        elif i2v_resolution == "540p":
            bucket_hw_base_size = 720
        elif i2v_resolution == "360p":
            bucket_hw_base_size = 480
        else:
            raise ValueError(f"i2v_resolution: {i2v_resolution} must be in [360p, 540p, 720p]")
        origin_size = semantic_images[0].size

        crop_size_list = self.generate_crop_size_list(bucket_hw_base_size, 32)
        aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
        closest_size, closest_ratio = self.get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)
        ref_image_transform = transforms.Compose([
            transforms.Resize(closest_size),
            transforms.CenterCrop(closest_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        semantic_image_pixel_values = [ref_image_transform(semantic_image) for semantic_image in semantic_images]
        semantic_image_pixel_values = torch.cat(semantic_image_pixel_values).unsqueeze(0).unsqueeze(2).to(self.device)
        target_height, target_width = closest_size
        return semantic_image_pixel_values, target_height, target_width


    def encode_prompt(self, prompt, positive=True, clip_sequence_length=77, llm_sequence_length=256, input_images=None):
        prompt_emb, pooled_prompt_emb, text_mask = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive, clip_sequence_length=clip_sequence_length, llm_sequence_length=llm_sequence_length, images=input_images
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_mask": text_mask}


    def prepare_extra_input(self, latents=None, guidance=1.0):
        freqs_cos, freqs_sin = self.dit.prepare_freqs(latents)
        guidance = torch.Tensor([guidance] * latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
        return {"freqs_cos": freqs_cos, "freqs_sin": freqs_sin, "guidance": guidance}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames


    def encode_video(self, frames, tile_size=(17, 30, 30), tile_stride=(12, 20, 20)):
        tile_size = ((tile_size[0] - 1) * 4 + 1, tile_size[1] * 8, tile_size[2] * 8)
        tile_stride = (tile_stride[0] * 4, tile_stride[1] * 8, tile_stride[2] * 8)
        latents = self.vae_encoder.encode_video(frames, tile_size=tile_size, tile_stride=tile_stride)
        return latents


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_video=None,
        input_images=None,
        i2v_resolution="720p",
        i2v_stability=True,
        denoising_strength=1.0,
        seed=None,
        rand_device=None,
        height=720,
        width=1280,
        num_frames=129,
        embedded_guidance=6.0,
        cfg_scale=1.0,
        num_inference_steps=30,
        tea_cache_l1_thresh=None,
        tile_size=(17, 30, 30),
        tile_stride=(12, 20, 20),
        step_processor=None,
        progress_bar_cmd=lambda x: x,
        progress_bar_st=None,
    ):
        # Tiler parameters
        tiler_kwargs = {"tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # encoder input images
        if input_images is not None:
            self.load_models_to_device(['vae_encoder'])
            image_pixel_values, height, width = self.prepare_vae_images_inputs(input_images, i2v_resolution=i2v_resolution)
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                image_latents = self.vae_encoder(image_pixel_values)

        # Initialize noise
        rand_device = self.device if rand_device is None else rand_device
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=self.torch_dtype).to(self.device)
        if input_video is not None:
            self.load_models_to_device(['vae_encoder'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        elif input_images is not None and i2v_stability:
            noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=image_latents.dtype).to(self.device)
            t = torch.tensor([0.999]).to(device=self.device)
            latents = noise * t + image_latents.repeat(1, 1, (num_frames - 1) // 4 + 1, 1, 1) * (1 - t)
            latents = latents.to(dtype=image_latents.dtype)
        else:
            latents = noise

        # Encode prompts
        # current mllm does not support vram_management
        self.load_models_to_device(["text_encoder_1"] if self.vram_management and input_images is None else ["text_encoder_1", "text_encoder_2"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True, input_images=input_images)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # TeaCache
        tea_cache_kwargs = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None}

        # Denoise
        self.load_models_to_device([] if self.vram_management else ["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)
            print(f"Step {progress_id + 1} / {len(self.scheduler.timesteps)}")

            forward_func = lets_dance_hunyuan_video
            if input_images is not None:
                latents = torch.concat([image_latents, latents[:, :, 1:, :, :]], dim=2)
                forward_func = lets_dance_hunyuan_video_i2v

            # Inference
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                noise_pred_posi = forward_func(self.dit, latents, timestep, **prompt_emb_posi, **extra_input, **tea_cache_kwargs)
                if cfg_scale != 1.0:
                    noise_pred_nega = forward_func(self.dit, latents, timestep, **prompt_emb_nega, **extra_input)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

            # (Experimental feature, may be removed in the future)
            if step_processor is not None:
                self.load_models_to_device(['vae_decoder'])
                rendered_frames = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents, to_final=True)
                rendered_frames = self.vae_decoder.decode_video(rendered_frames, **tiler_kwargs)
                rendered_frames = self.tensor2video(rendered_frames[0])
                rendered_frames = step_processor(rendered_frames, original_frames=input_video)
                self.load_models_to_device(['vae_encoder'])
                rendered_frames = self.preprocess_images(rendered_frames)
                rendered_frames = torch.stack(rendered_frames, dim=2)
                target_latents = self.encode_video(rendered_frames).to(dtype=self.torch_dtype, device=self.device)
                noise_pred = self.scheduler.return_to_timestep(self.scheduler.timesteps[progress_id], latents, target_latents)
                self.load_models_to_device([] if self.vram_management else ["dit"])

            # Scheduler
            if input_images is not None:
                latents = self.scheduler.step(noise_pred[:, :, 1:, :, :], self.scheduler.timesteps[progress_id], latents[:, :, 1:, :, :])
                latents = torch.concat([image_latents, latents], dim=2)
            else:
                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        # Decode
        self.load_models_to_device(['vae_decoder'])
        frames = self.vae_decoder.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

    def check(self, dit: HunyuanVideoDiT, img, vec):
        img_ = img.clone()
        vec_ = vec.clone()
        img_mod1_shift, img_mod1_scale, _, _, _, _ = dit.double_blocks[0].component_a.mod(vec_).chunk(6, dim=-1)
        normed_inp = dit.double_blocks[0].component_a.norm1(img_)
        modulated_inp = normed_inp * (1 + img_mod1_scale.unsqueeze(1)) + img_mod1_shift.unsqueeze(1)
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = [7.33226126e+02, -4.01131952e+02,  6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = img.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def lets_dance_hunyuan_video(
    dit: HunyuanVideoDiT,
    x: torch.Tensor,
    t: torch.Tensor,
    prompt_emb: torch.Tensor = None,
    text_mask: torch.Tensor = None,
    pooled_prompt_emb: torch.Tensor = None,
    freqs_cos: torch.Tensor = None,
    freqs_sin: torch.Tensor = None,
    guidance: torch.Tensor = None,
    tea_cache: TeaCache = None,
    **kwargs
):
    B, C, T, H, W = x.shape

    vec = dit.time_in(t, dtype=torch.float32) + dit.vector_in(pooled_prompt_emb) + dit.guidance_in(guidance * 1000, dtype=torch.float32)
    img = dit.img_in(x)
    txt = dit.txt_in(prompt_emb, t, text_mask)

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, img, vec)
    else:
        tea_cache_update = False

    if tea_cache_update:
        print("TeaCache skip forward.")
        img = tea_cache.update(img)
    else:
        split_token = int(text_mask.sum(dim=1))
        txt_len = int(txt.shape[1])
        for block in tqdm(dit.double_blocks, desc="Double stream blocks"):
            img, txt = block(img, txt, vec, (freqs_cos, freqs_sin), split_token=split_token)

        x = torch.concat([img, txt], dim=1)
        for block in tqdm(dit.single_blocks, desc="Single stream blocks"):
            x = block(x, vec, (freqs_cos, freqs_sin), txt_len=txt_len, split_token=split_token)
        img = x[:, :-txt_len]

        if tea_cache is not None:
            tea_cache.store(img)
    img = dit.final_layer(img, vec)
    img = dit.unpatchify(img, T=T//1, H=H//2, W=W//2)
    return img


def lets_dance_hunyuan_video_i2v(
    dit: HunyuanVideoDiT,
    x: torch.Tensor,
    t: torch.Tensor,
    prompt_emb: torch.Tensor = None,
    text_mask: torch.Tensor = None,
    pooled_prompt_emb: torch.Tensor = None,
    freqs_cos: torch.Tensor = None,
    freqs_sin: torch.Tensor = None,
    guidance: torch.Tensor = None,
    tea_cache: TeaCache = None,
    **kwargs
):
    B, C, T, H, W = x.shape
    # Uncomment below to keep same as official implementation
    # guidance = guidance.to(dtype=torch.float32).to(torch.bfloat16)
    vec = dit.time_in(t, dtype=torch.bfloat16)
    vec_2 = dit.vector_in(pooled_prompt_emb)
    vec = vec + vec_2
    vec = vec + dit.guidance_in(guidance * 1000., dtype=torch.bfloat16)

    token_replace_vec = dit.time_in(torch.zeros_like(t), dtype=torch.bfloat16)
    tr_token = (H // 2) * (W // 2)
    token_replace_vec = token_replace_vec + vec_2

    img = dit.img_in(x)
    txt = dit.txt_in(prompt_emb, t, text_mask)

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, img, vec)
    else:
        tea_cache_update = False

    if tea_cache_update:
        print("TeaCache skip forward.")
        img = tea_cache.update(img)
    else:
        split_token = int(text_mask.sum(dim=1))
        txt_len = int(txt.shape[1])
        for block in tqdm(dit.double_blocks, desc="Double stream blocks"):
            img, txt = block(img, txt, vec, (freqs_cos, freqs_sin), token_replace_vec, tr_token, split_token)

        x = torch.concat([img, txt], dim=1)
        for block in tqdm(dit.single_blocks, desc="Single stream blocks"):
            x = block(x, vec, (freqs_cos, freqs_sin), txt_len, token_replace_vec, tr_token, split_token)
        img = x[:, :-txt_len]

        if tea_cache is not None:
            tea_cache.store(img)
    img = dit.final_layer(img, vec)
    img = dit.unpatchify(img, T=T//1, H=H//2, W=W//2)
    return img
