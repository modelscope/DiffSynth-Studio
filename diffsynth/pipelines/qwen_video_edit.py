import torch, math
from typing import Union
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.qwen_image_dit import QwenImageDiT
from ..models.qwen_image_text_encoder import QwenImageTextEncoder
from ..models.qwen_video_edit_dit import QwenVideoEditAdapter, QwenVideoEditRope
from ..models.wan_video_vae import WanVideoVAE


class QwenVideoEditPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
            time_division_factor=4, time_division_remainder=1,
        )
        from transformers import Qwen2Tokenizer, Qwen2VLProcessor

        self.scheduler = FlowMatchScheduler("Qwen-Image")
        self.text_encoder: QwenImageTextEncoder = None
        self.dit: QwenImageDiT = None
        self.vae: WanVideoVAE = None
        self.tokenizer: Qwen2Tokenizer = None
        self.processor: Qwen2VLProcessor = None
        self.adapter: QwenVideoEditAdapter = None
        self.in_iteration_models = ("dit", "adapter")
        self.units = [
            QwenVideoEditUnit_EditVideoEmbedder(),
            QwenVideoEditUnit_NoiseInitializer(),
            QwenVideoEditUnit_InputVideoEmbedder(),
            QwenVideoEditUnit_PromptEmbedder(),
        ]
        self.model_fn = model_fn_qwen_video_edit
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = QwenVideoEditPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_pool.fetch_model("qwen_video_edit_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.adapter = model_pool.fetch_model("qwen_video_edit_adapter")

        # Size division factor derived from VAE
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen2VLProcessor
            pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)

        # Grid-aware RoPE
        origin_rope = pipe.dit.pos_embed
        if not isinstance(origin_rope, QwenVideoEditRope):
            pipe.dit.pos_embed = QwenVideoEditRope(
                theta=origin_rope.theta, axes_dim=origin_rope.axes_dim, scale_rope=origin_rope.scale_rope,
            ).to(device)

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Video
        edit_video: list[Image.Image] = None,
        input_video: list[Image.Image] = None,
        num_frames: int = 45,
        height: int = 384,
        width: int = 640,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        # Prompt
        prompts: list[str] = [],
        negative_prompt: str = " ",
        cfg_scale: float = 4.0,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 40,
        # Qwen-Video-Edit
        zero_cond_t: bool = False,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        # Shape check
        height, width, num_frames = self.check_resize_height_width(height, width, num_frames)

        # Scheduler
        num_groups = ((num_frames - 1) // 4 + 1) // self.adapter.in_proj.group
        self.scheduler.set_timesteps(num_inference_steps, dynamic_shift_len=num_groups * (height // self.height_division_factor) * (width // self.width_division_factor))

        total_frames = len(edit_video)
        num_video_chunks = (total_frames + num_frames - 1) // num_frames
        num_chunks = min(len(prompts), num_video_chunks)
        if len(prompts) < num_video_chunks:
            print(f"Warning: only {len(prompts)} prompts provided for {num_video_chunks} chunks. "
                  f"The last {total_frames - num_chunks * num_frames} frames will be dropped.")

        videos = []
        for chunk_id in range(num_chunks):
            inputs_posi = {"prompt": prompts[chunk_id]}
            inputs_nega = {"negative_prompt": negative_prompt}
            inputs_shared = {
                "edit_video": edit_video,
                "input_video": input_video,
                "chunk_id": chunk_id,
                "num_frames": num_frames,
                "height": height, "width": width,
                "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
                "cfg_scale": cfg_scale,
                "seed": seed, "rand_device": rand_device,
                "zero_cond_t": zero_cond_t,
            }
            for unit in self.units:
                inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

            # Denoise
            self.load_models_to_device(self.in_iteration_models)
            models = {name: getattr(self, name) for name in self.in_iteration_models}
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
                if cfg_scale != 1.0:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                    noise_pred = noise_pred * (torch.norm(noise_pred_posi, dim=1, keepdim=True) / torch.norm(noise_pred, dim=1, keepdim=True).clamp_min(1e-6))
                else:
                    noise_pred = noise_pred_posi
                inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

            # Decode
            self.load_models_to_device(["vae"])
            edited = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].cpu()
            num_valid_frames = inputs_shared["num_valid_frames"]
            edited = edited[:, :num_valid_frames]
            videos.append(edited)

        self.load_models_to_device([])
        output_video = torch.cat(videos, dim=1).unsqueeze(0)
        return self.vae_output_to_video(output_video, pattern="B C T H W", min_value=-1, max_value=1)


class QwenVideoEditUnit_EditVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_video", "chunk_id", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("edit_video_chunk", "num_valid_frames", "ref_latents"),
            onload_model_names=("vae",),
        )

    @staticmethod
    def encode_video_chunk(pipe: QwenVideoEditPipeline, video, chunk_id, num_frames, height, width, tiled, tile_size, tile_stride):
        start = chunk_id * num_frames
        end = min(start + num_frames, len(video))
        frames = [video[i].resize((width, height)) for i in range(start, end)]
        padded_frames = frames + [frames[-1]] * (num_frames - len(frames))
        pipe.load_models_to_device(("vae",))
        latents = pipe.vae.encode(
            pipe.preprocess_video(padded_frames), device=pipe.device,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        return frames, latents

    def process(self, pipe: QwenVideoEditPipeline, edit_video, chunk_id, num_frames, height, width, tiled, tile_size, tile_stride):
        frames, ref_latents = self.encode_video_chunk(pipe, edit_video, chunk_id, num_frames, height, width, tiled, tile_size, tile_stride)
        return {"edit_video_chunk": frames, "num_valid_frames": len(frames), "ref_latents": ref_latents}


class QwenVideoEditUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "chunk_id", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: QwenVideoEditPipeline, input_video, noise, chunk_id, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_video is None or not pipe.scheduler.training:
            return {}
        _, input_latents = QwenVideoEditUnit_EditVideoEmbedder.encode_video_chunk(
            pipe, input_video, chunk_id, num_frames, height, width, tiled, tile_size, tile_stride)
        return {"latents": noise, "input_latents": input_latents}


class QwenVideoEditUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("ref_latents", "seed", "rand_device"),
            output_params=("noise", "latents"),
        )

    def process(self, pipe: QwenVideoEditPipeline, ref_latents, seed, rand_device):
        noise = pipe.generate_noise(
            ref_latents.shape, seed=seed, rand_device=rand_device, rand_torch_dtype=torch.float32,
            device=pipe.device, torch_dtype=pipe.torch_dtype,
        )
        if pipe.scheduler.training:
            return {"noise": noise}
        return {"noise": noise, "latents": noise}


class QwenVideoEditUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_video_chunk",),
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",),
        )

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def build_preview_grid(self, frames, rows=3, cols=3, target_area=1024 * 1024):
        sample_indices = np.linspace(0, len(frames) - 1, rows * cols).round().astype(int)
        tiles = [frames[i] for i in sample_indices]
        tile_w, tile_h = tiles[0].size
        scale = (target_area / (tile_w * cols * tile_h * rows)) ** 0.5
        grid_tile_w = max(int(tile_w * scale) // 2 * 2, 2)
        grid_tile_h = max(int(tile_h * scale) // 2 * 2, 2)
        grid = Image.new("RGB", (grid_tile_w * cols, grid_tile_h * rows))
        for i, tile in enumerate(tiles):
            grid.paste(tile.resize((grid_tile_w, grid_tile_h)), ((i % cols) * grid_tile_w, (i // cols) * grid_tile_h))
        return grid

    def encode_prompt_edit(self, pipe: QwenVideoEditPipeline, prompt, edit_image):
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states

    def process(self, pipe: QwenVideoEditPipeline, prompt, edit_video_chunk) -> dict:
        if pipe.text_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        preview_image = self.build_preview_grid(edit_video_chunk)
        split_hidden_states = self.encode_prompt_edit(pipe, [prompt], preview_image)
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
        encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
        prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask}


def factorize_grid(value):
    for rows in range(int(math.sqrt(value)), 0, -1):
        if value % rows == 0:
            return rows, value // rows
    return 1, value


def model_fn_qwen_video_edit(
    dit: QwenImageDiT = None,
    adapter: QwenVideoEditAdapter = None,
    latents: torch.Tensor = None,
    ref_latents: torch.Tensor = None,
    prompt_emb: torch.Tensor = None,
    prompt_emb_mask: torch.Tensor = None,
    timestep: torch.Tensor = None,
    zero_cond_t: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    in_proj, out_proj = adapter.in_proj, adapter.out_proj
    _, _, num_latent_frames, latent_height, latent_width = latents.shape
    num_groups = num_latent_frames // in_proj.group
    tokens_h, tokens_w = latent_height // 2, latent_width // 2
    rows, cols = factorize_grid(num_groups)
    img_shapes = []

    for frame_index in (0, 1):  # frame axis: 0 for noise latents, 1 for reference latents
        for group_index in range(num_groups):
            img_shapes.append({
                "frame": frame_index,
                "height": tokens_h,
                "width": tokens_w,
                "h_off": (group_index // cols) * tokens_h,
                "w_off": (group_index % cols) * tokens_w,
                "full_height": rows * tokens_h,
                "full_width": cols * tokens_w,
            })

    image = torch.cat([in_proj(latents), in_proj(ref_latents)], dim=1)
    image_seq_len = image.shape[1] // 2
    timestep = timestep / 1000

    if zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        noise_len = sum(item["height"] * item["width"] for item in img_shapes[:num_groups])
        cond_len = sum(item["height"] * item["width"] for item in img_shapes[num_groups:])
        modulate_index = torch.tensor([[0] * noise_len + [1] * cond_len], device=image.device, dtype=torch.int)
    else:
        modulate_index = None

    conditioning = dit.time_text_embed(
        timestep, image.dtype,
        addition_t_cond=None if not dit.time_text_embed.use_additional_t_cond else
        torch.tensor([0]).to(device=image.device, dtype=torch.long),
    )
    text = dit.txt_in(dit.txt_norm(prompt_emb))
    image_rotary_emb = dit.pos_embed(img_shapes, prompt_emb_mask.sum(dim=1).tolist(), device=image.device)

    for block in dit.transformer_blocks:
        text, image = gradient_checkpoint_forward(
            block, use_gradient_checkpointing, use_gradient_checkpointing_offload,
            image=image, text=text, temb=conditioning,
            image_rotary_emb=image_rotary_emb, attention_mask=None, modulate_index=modulate_index,
        )

    if zero_cond_t:
        conditioning = conditioning.chunk(2, dim=0)[0]
    image = dit.norm_out(image, conditioning)[:, :image_seq_len]
    output = out_proj(image, num_groups, tokens_h, tokens_w)
    return output
