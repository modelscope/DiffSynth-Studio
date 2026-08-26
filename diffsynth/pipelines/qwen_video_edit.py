import torch, math
from typing import Union
from einops import rearrange
import numpy as np
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

from ..core import ModelConfig, gradient_checkpoint_forward
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion.base_pipeline import BasePipeline
from ..models.qwen_image_dit import QwenEmbedRope
from .qwen_image import QwenImageUnit_PromptEmbedder
from ..diffusion import FlowMatchScheduler


class QwenVideoEditRope(QwenEmbedRope):
    def _expand_pos_freqs_if_needed(self, video_fhw, txt_seq_lens):
        if isinstance(video_fhw, list) and video_fhw and isinstance(video_fhw[0], dict):
            video_fhw = (
                max(x["frame"] + 1 for x in video_fhw),
                max(x["full_height"] for x in video_fhw),
                max(x["full_width"] for x in video_fhw),
            )
        super()._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)

    def forward(self, video_fhw, txt_seq_lens, device):
        if not video_fhw or not isinstance(video_fhw[0], dict):
            return super().forward(video_fhw, txt_seq_lens, device)
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)

        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        video_freqs, max_index = [], 0
        for item in video_fhw:
            frame, height, width = item["frame"], item["height"], item["width"]
            full_height, full_width = item["full_height"], item["full_width"]
            key = "grid_" + "_".join(str(item[x]) for x in ("frame", "height", "width", "h_off", "w_off", "full_height", "full_width"))
            if key not in self.rope_cache:
                frame_freq = freqs_pos[0][frame:frame + 1].view(1, 1, 1, -1).expand(1, height, width, -1)
                if self.scale_rope:
                    axis_h = torch.cat([freqs_neg[1][-(full_height - full_height // 2):], freqs_pos[1][:full_height // 2]])
                    axis_w = torch.cat([freqs_neg[2][-(full_width - full_width // 2):], freqs_pos[2][:full_width // 2]])
                else:
                    axis_h, axis_w = freqs_pos[1][:full_height], freqs_pos[2][:full_width]
                height_freq = axis_h[item["h_off"]:item["h_off"] + height].view(1, height, 1, -1).expand(1, height, width, -1)
                width_freq = axis_w[item["w_off"]:item["w_off"] + width].view(1, 1, width, -1).expand(1, height, width, -1)
                self.rope_cache[key] = torch.cat([frame_freq, height_freq, width_freq], dim=-1).reshape(height * width, -1).contiguous()
            video_freqs.append(self.rope_cache[key])
            max_index = max(full_height // 2, full_width // 2, max_index) if self.scale_rope else max(full_height, full_width, max_index)
        
        text_freqs = self.pos_freqs[max_index:max_index + max(txt_seq_lens)]
        return torch.cat(video_freqs, dim=0), text_freqs


class WanToQwenProjection(torch.nn.Module):
    def __init__(self, in_channels: int = 16, inner_dim: int = 3072):
        super().__init__()
        self.group = 1
        self.proj = torch.nn.Conv3d(in_channels, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    @torch.no_grad()
    def init_from_qwen_dit(self, dit):
        self.proj.weight.copy_(dit.img_in.weight.view(self.proj.out_channels, 16, 2, 2).unsqueeze(2))
        self.proj.bias.copy_(dit.img_in.bias)

    def forward(self, x):
        return rearrange(self.proj(x), "B D T H W -> B (T H W) D")


class QwenToWanProjection(torch.nn.Module):
    def __init__(self, out_channels: int = 16, inner_dim: int = 3072):
        super().__init__()
        self.group = 1
        self.proj = torch.nn.Linear(inner_dim, out_channels * 4)

    @torch.no_grad()
    def init_from_qwen_dit(self, dit):
        self.proj.load_state_dict(dit.proj_out.state_dict())

    def forward(self, x, num_frames, tokens_h, tokens_w):
        return rearrange(self.proj(x), "B (T H W) (C P Q) -> B C T (H P) (W Q)",
                         T=num_frames, H=tokens_h, W=tokens_w, P=2, Q=2)


def _factorize(value):
    for rows in range(int(math.sqrt(value)), 0, -1):
        if value % rows == 0:
            return rows, value // rows
    return 1, value


class QwenVideoEditPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        from transformers import Qwen2Tokenizer, Qwen2VLProcessor

        self.scheduler = FlowMatchScheduler("Qwen-Image")
        self.text_encoder: QwenImageTextEncoder = None
        self.dit: QwenImageDiT = None
        self.video_vae = None
        self.tokenizer: Qwen2Tokenizer = None
        self.processor: Qwen2VLProcessor = None
        self.in_proj: WanToQwenProjection = None
        self.out_proj: QwenToWanProjection = None
        self.prompt_embedder = QwenImageUnit_PromptEmbedder()
        self.in_iteration_models = ("dit", "in_proj", "out_proj")
        self.units = []
        self.model_fn = model_fn_qwen_video_edit
        self.compilable_models = ["dit"]


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        video_vae_config: ModelConfig = None,
        checkpoint: ModelConfig = None,
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = QwenVideoEditPipeline(device=device, torch_dtype=torch_dtype)
        configs = list(model_configs)
        if video_vae_config is not None:
            configs.append(video_vae_config)
        model_pool = pipe.download_and_load_models(configs, vram_limit)

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_pool.fetch_model("qwen_image_dit")
        pipe.video_vae = model_pool.fetch_model("wan_video_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen2VLProcessor
            pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)

        # Replace RoPE with the mosaic-aware variant
        origin_rope = pipe.dit.pos_embed
        pipe.dit.pos_embed = QwenVideoEditRope(
            theta=origin_rope.theta, axes_dim=origin_rope.axes_dim,
            scale_rope=origin_rope.scale_rope,
        ).to(device)

        # Wan latent <-> Qwen token projections
        inner_dim = pipe.dit.img_in.out_features
        pipe.in_proj = WanToQwenProjection(in_channels=16, inner_dim=inner_dim).to(device, torch_dtype)
        pipe.out_proj = QwenToWanProjection(out_channels=16, inner_dim=inner_dim).to(device, torch_dtype)
        pipe.in_proj.init_from_qwen_dit(pipe.dit)
        pipe.out_proj.init_from_qwen_dit(pipe.dit)

        # Load Fine-tuned weights
        if checkpoint is not None:
            checkpoint.download_if_necessary()
            state = load_file(checkpoint.path)
            pipe.in_proj.load_state_dict({k[len("in_proj."):]: v for k, v in state.items() if k.startswith("in_proj.")})
            pipe.out_proj.load_state_dict({k[len("out_proj."):]: v for k, v in state.items() if k.startswith("out_proj.")})
            dit_state = {k[len("pipe.dit."):]: v for k, v in state.items() if k.startswith("pipe.dit.")}
            if dit_state:
                if any("lora" in key for key in dit_state):
                    pipe.load_lora(pipe.dit, state_dict=dit_state, hotload=True)
                else:
                    pipe.dit.load_state_dict(dit_state, strict=False)

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @staticmethod
    def build_preview_grid(frames, rows=3, cols=3, target_area=1024 * 1024):
        """Uniformly sampled frames tiled into one grid image -- the Qwen2.5-VL
        image prompt (the VL branch sees the whole video-as-grid)."""
        idx = np.linspace(0, len(frames) - 1, rows * cols).round().astype(int)
        tiles = [frames[i] for i in idx]
        w0, h0 = tiles[0].size
        scale = (target_area / (w0 * cols * h0 * rows)) ** 0.5
        tw, th = max(int(w0 * scale) // 2 * 2, 2), max(int(h0 * scale) // 2 * 2, 2)
        grid = Image.new("RGB", (tw * cols, th * rows))
        for i, tile in enumerate(tiles):
            grid.paste(tile.resize((tw, th), Image.BILINEAR), ((i % cols) * tw, (i // cols) * th))
        return grid

    @torch.no_grad()
    def __call__(
        self,
        # Video
        input_video: list[Image.Image] = None,
        num_frames: int = 45,
        max_pixels: int = 245760,
        tiled: bool = None,
        preview_image: Image.Image = None,
        # Prompt
        prompts: list[str] = [],
        negative_prompt: str = " ",
        cfg_scale: float = 4.0,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 40,
        denoising_strength: float = 1.0,
        # Qwen-Video-Edit
        zero_cond_t: bool = False,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        """Edit a long video with one prompt per chunk.

        Args:
            input_video: list of PIL.Image frames (the full source video).
            prompts: list of strings, one per chunk.
            num_frames: frames per chunk (must match training: 45).
        """
        # Resolve spatial dimensions
        w0, h0 = input_video[0].size
        scale = min(1.0, (max_pixels / (w0 * h0)) ** 0.5)
        height = max(round(h0 * scale / 16), 1) * 16
        width = max(round(w0 * scale / 16), 1) * 16

        if (w0, h0) != (width, height):
            input_video = [frame.resize((width, height), Image.BILINEAR) for frame in input_video]
        video = self.preprocess_video(input_video, torch_dtype=self.torch_dtype, device=self.device)

        total_frames = video.shape[2]
        n_chunks = max(1, (total_frames + num_frames - 1) // num_frames)
        results = []

        for cid in range(n_chunks):
            start = cid * num_frames
            end = min(start + num_frames, total_frames)
            chunk = video[:, :, start:end]
            if chunk.shape[2] < num_frames:
                pad = video[:, :, -1:].expand(1, chunk.shape[1], num_frames - chunk.shape[2], chunk.shape[3], chunk.shape[4])
                chunk = torch.cat([chunk, pad], dim=2)
            prompt = prompts[cid] if cid < len(prompts) else prompts[-1]
            height_chunk, width_chunk = chunk.shape[-2:]
            encode_tiled = tiled if tiled is not None else (height_chunk * width_chunk >= 700_000)

            self.load_models_to_device(["video_vae"])
            ref = self.video_vae.encode([chunk[0]], device=self.device, tiled=encode_tiled).to(dtype=self.torch_dtype, device=self.device)

            if preview_image is None:
                chunk_frames = [input_video[min(start + t, total_frames - 1)] for t in range(min(num_frames, end - start))]
                preview = self.build_preview_grid(chunk_frames)
            else:
                preview = preview_image
            emb = self.prompt_embedder.process(self, prompt=prompt, edit_image=preview)
            neg_emb = self.prompt_embedder.process(self, prompt=negative_prompt, edit_image=preview) if cfg_scale > 1 else None

            # Scheduler
            group = getattr(self.in_proj, "group", 1)
            noise_seq_len = (ref.shape[2] // group) * (ref.shape[3] // 2) * (ref.shape[4] // 2)
            self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=noise_seq_len)

            # Denoise
            self.load_models_to_device(self.in_iteration_models)
            latents = self.generate_noise(
                ref.shape, seed=seed, rand_device=rand_device, rand_torch_dtype=torch.float32,
                device=self.device, torch_dtype=self.torch_dtype,
            )
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps, desc=f"chunk {cid}")):
                timestep = timestep[None].to(self.device, self.torch_dtype)
                pred = self.model_fn(
                    self.dit, self.in_proj, self.out_proj, latents, ref,
                    emb["prompt_emb"], emb["prompt_emb_mask"], timestep,
                    zero_cond_t=zero_cond_t,
                )
                if neg_emb is not None:
                    neg_pred = self.model_fn(
                        self.dit, self.in_proj, self.out_proj, latents, ref,
                        neg_emb["prompt_emb"], neg_emb["prompt_emb_mask"], timestep,
                        zero_cond_t=zero_cond_t,
                    )
                    combined = neg_pred + cfg_scale * (pred - neg_pred)
                    pred = combined * (torch.norm(pred, dim=1, keepdim=True) /
                                       torch.norm(combined, dim=1, keepdim=True).clamp_min(1e-6))
                latents = self.step(self.scheduler, latents=latents, progress_id=progress_id, noise_pred=pred)

            # Decode
            decode_tiled = tiled if tiled is not None else (latents.shape[3] * 8) * (latents.shape[4] * 8) >= 700_000
            self.load_models_to_device(["video_vae"])
            edited = self.video_vae.decode(latents, device=self.device, tiled=decode_tiled)[0].cpu()
            actual = end - start
            edited = edited[:, :actual] if cid == n_chunks - 1 and actual < num_frames else edited
            results.append(edited)

        self.load_models_to_device([])
        video = torch.cat(results, dim=1).unsqueeze(0)
        return self.vae_output_to_video(video, pattern="B C T H W", min_value=-1, max_value=1)


def model_fn_qwen_video_edit(
    dit, in_proj, out_proj, 
    latents, ref_latents, prompt_emb, prompt_mask, timestep,
    zero_cond_t=False,
):
    _, _, frames, height, width = latents.shape
    groups, tokens_h, tokens_w = frames // in_proj.group, height // 2, width // 2
    rows, cols = _factorize(groups)
    shapes = []

    for base in (0, 1):
        shapes.extend({"frame": base, "height": tokens_h, "width": tokens_w,
                       "h_off": (i // cols) * tokens_h, "w_off": (i % cols) * tokens_w,
                       "full_height": rows * tokens_h, "full_width": cols * tokens_w} for i in range(groups))
    image = torch.cat([in_proj(latents), in_proj(ref_latents)], dim=1)
    image_len = image.shape[1] // 2
    timestep = timestep / 1000

    if zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        noise_len = sum(item["height"] * item["width"] for item in shapes[:groups])
        cond_len = sum(item["height"] * item["width"] for item in shapes[groups:])
        modulate_index = torch.tensor([[0] * noise_len + [1] * cond_len], device=image.device, dtype=torch.int)
    else:
        modulate_index = None

    conditioning = dit.time_text_embed(
        timestep, image.dtype,
        addition_t_cond=None if not dit.time_text_embed.use_additional_t_cond else
        torch.tensor([0], device=image.device, dtype=torch.long),)
    text = dit.txt_in(dit.txt_norm(prompt_emb))
    rotary = dit.pos_embed(shapes, prompt_mask.sum(dim=1).tolist(), device=image.device)

    for block in dit.transformer_blocks:
        text, image = gradient_checkpoint_forward(
            block, False, False, image=image, text=text, temb=conditioning,
            image_rotary_emb=rotary, attention_mask=None, modulate_index=modulate_index)

    if zero_cond_t:
        conditioning = conditioning.chunk(2, dim=0)[0]
    image = dit.norm_out(image, conditioning)[:, :image_len]
    return out_proj(image, groups, tokens_h, tokens_w)
