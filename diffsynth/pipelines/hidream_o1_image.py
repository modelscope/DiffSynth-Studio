# HiDream-O1-Image Pipeline for DiffSynth-Studio.

import torch, einops
from typing import Optional, Union
from tqdm import tqdm
from PIL import Image
import numpy as np

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.hidream_o1_image_dit import HiDreamO1ImageModel
from ..models.hidream_common import add_special_tokens, get_rope_index_fix_point
from transformers import AutoTokenizer


# ──────────────────────────────────────────────────────────────────────
# Flash scheduler step (temporary, matches FlashFlowMatchEulerDiscreteScheduler.step)
# ──────────────────────────────────────────────────────────────────────
def _flash_step(sigmas, step_index, model_output, sample, s_noise=1.0, noise_clip_std=0.0):
    """Equivalent to FlashFlowMatchEulerDiscreteScheduler.step (flash_scheduler.py L276-362)."""
    sigma = sigmas[step_index]
    sample = sample.to(torch.float32)
    model_output = model_output.to(torch.float32)

    denoised = sample - model_output * sigma

    if step_index < len(sigmas) - 1:
        sigma_next = sigmas[step_index + 1]
        noise = torch.randn_like(denoised)
        if noise_clip_std > 0:
            noise_std = noise.std().item()
            clip_val = noise_clip_std * noise_std
            noise = noise.clamp(min=-clip_val, max=clip_val)
        sample = sigma_next * noise * s_noise + (1.0 - sigma_next) * denoised
    else:
        sample = denoised

    return sample


DEFAULT_TIMESTEPS = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
]

TIMESTEP_TOKEN_NUM = 1
PATCH_SIZE = 32


def build_text_sample(prompt, height, width, tokenizer, processor, model_config, device):
    """Equivalent to target library's build_t2i_text_sample (pipeline.py L30-77)."""
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    messages = [{"role": "user", "content": prompt}]
    template_caption = (
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + boi_token
        + tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

    image_grid_thw = torch.tensor(
        [1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64
    ).unsqueeze(0)

    vision_tokens = torch.zeros((1, image_len), dtype=input_ids.dtype) + image_token_id
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids = _get_rope_index(
        image_token_id, video_token_id, vision_start_token_id,
        input_ids=input_ids_pad, image_grid_thw=image_grid_thw,
        device=device,
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, bgn: bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3

    vinput_mask = (token_types == 1)
    token_types_bin = (token_types > 0).to(token_types.dtype)

    return {
        'input_ids': input_ids.to(device),
        'position_ids': position_ids.to(device),
        'token_types': token_types_bin.to(device),
        'vinput_mask': vinput_mask.to(device),
    }


def _get_rope_index(image_token_id, video_token_id, vision_start_token_id, input_ids, image_grid_thw, device):
    """Rope index for Text-to-Image: matches target library's get_rope_index_fix_point with spatial_merge_size=1."""
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    spatial_merge_size = 1  # Target library passes spatial_merge_size=1 for T2I
    skip_vision_start_token = [1]
    fix_point = 4096

    position_ids = torch.ones(3, batch_size, seq_len, dtype=input_ids.dtype, device=device)

    for i in range(batch_size):
        input_ids_i = input_ids[i]

        # Find first image_token_id to determine text boundary (matches TL: ed_image = input_tokens.index(image_token_id, st))
        first_image_idx = (input_ids_i == image_token_id).nonzero(as_tuple=True)[0][0].item()
        text_len = first_image_idx - skip_vision_start_token[0]  # exclude vision_start_token from text positions

        # Build position IDs sequentially: text → vision → remaining text
        llm_pos_ids_list = []
        st_idx = 0

        # Text positions before vision
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        # Vision positions: 3D grid with fix_point offset
        t, h, w = image_grid_thw[i][0].item(), image_grid_thw[i][1].item(), image_grid_thw[i][2].item()
        llm_grid_t, llm_grid_h, llm_grid_w = t, h // spatial_merge_size, w // spatial_merge_size

        # fix_point is adjusted by subtracting st_idx for the first image (TL line 162-163)
        fix_point_adj = fix_point - st_idx
        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fix_point_adj + st_idx)

        # Remaining text after vision tokens (if any)
        num_vision_tokens = llm_grid_t * llm_grid_h * llm_grid_w
        st = first_image_idx + num_vision_tokens
        if st < seq_len:
            remaining_len = seq_len - st
            st_idx = llm_pos_ids_list[-1].max() + 1
            llm_pos_ids_list.append(torch.arange(remaining_len).view(1, -1).expand(3, -1) + st_idx)

        # Concatenate and assign
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, :] = llm_positions[..., :seq_len]

    return position_ids


# ──────────────────────────────────────────────────────────────────────
# Pipeline Units
# ──────────────────────────────────────────────────────────────────────
class HiDreamO1ImageUnit_PromptTokenizer(PipelineUnit):
    """Tokenize prompt into input_ids, position_ids, token_types, vinput_mask."""

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=("tokenizer", "processor", "model_config", "height", "width", "device"),
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"negative_prompt": "negative_prompt"},
            output_params=("input_ids", "position_ids", "token_types", "vinput_mask"),
        )

    def process(self, pipe, prompt=None, negative_prompt=None, **kwargs):
        from ..pipelines.hidream_o1_image import build_text_sample

        tokenizer = kwargs.get("tokenizer") or pipe.tokenizer
        processor = kwargs.get("processor") or pipe.processor
        model_config = kwargs.get("model_config") or pipe.dit.config
        height = kwargs.get("height")
        width = kwargs.get("width")
        device = kwargs.get("device")

        text = prompt if prompt is not None else negative_prompt
        result = build_text_sample(
            prompt=text,
            height=height, width=width,
            tokenizer=tokenizer, processor=processor,
            model_config=model_config, device=device,
        )
        return result


class HiDreamO1ImageUnit_NoiseInitializer(PipelineUnit):
    """Generate pixel-level noise and rearrange to patch space."""

    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "dtype", "device", "noise_scale_start"),
            output_params=("latents",),
        )

    def prepare_inputs(self, inputs_shared, inputs_posi, inputs_nega):
        return inputs_shared, inputs_posi, inputs_nega

    def process(self, pipe, height=None, width=None, seed=None, **kwargs):
        dtype = kwargs.get("dtype")
        device = kwargs.get("device")
        noise_scale = kwargs.get("noise_scale_start", 7.5)

        generator = torch.Generator('cpu')
        if seed is not None:
            generator.manual_seed(seed + 1)  # target library uses seed+1

        noise = noise_scale * torch.randn(
            (1, 3, height, width), generator=generator, dtype=torch.float32
        ).to(device=device, dtype=dtype)

        patch_size = 32
        latents = einops.rearrange(
            noise, 'B C (H p1) (W p2) -> B (H W) (C p1 p2)',
            p1=patch_size, p2=patch_size,
        )

        return {"latents": latents}


# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────

class HiDreamO1ImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=32, width_division_factor=32,
        )
        self.scheduler = FlowMatchScheduler("HiDream-O1-Image")
        self.dit: HiDreamO1ImageModel = None
        self.tokenizer = None
        self.processor = None

        self.in_iteration_models = ("dit",)
        self.units = [
            HiDreamO1ImageUnit_PromptTokenizer(),
            HiDreamO1ImageUnit_NoiseInitializer(),
        ]
        self.model_fn = model_fn_hidream_o1_image

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="HiDream-ai/HiDream-O1-Image-Dev", origin_file_pattern="./"),
        processor_config: ModelConfig = ModelConfig(model_id="HiDream-ai/HiDream-O1-Image-Dev", origin_file_pattern="./"),
        vram_limit: float = None,
    ):
        pipe = HiDreamO1ImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.dit = model_pool.fetch_model("hidream_o1_image_dit")
        if tokenizer_config is not None:
            from transformers import AutoTokenizer, AutoProcessor
            tokenizer_config.download_if_necessary()
            processor_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = " ",
        cfg_scale: float = 0.0,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        num_inference_steps: int = 28,
        shift: float = 1.0,
        noise_scale_start: float = 7.5,
        noise_scale_end: float = 7.5,
        noise_clip_std: float = 2.5,
        progress_bar_cmd=tqdm,
    ):
        device = self.device
        dtype = self.torch_dtype

        # 1. Scheduler: set timesteps for Dev mode
        self.scheduler.set_timesteps(
            num_inference_steps, shift=shift,
            special_case="flash", timesteps_list=DEFAULT_TIMESTEPS,
        )

        # Noise scale schedule
        num_steps = len(self.scheduler.timesteps)
        if num_steps > 1:
            noise_scale_schedule = [
                noise_scale_start + (noise_scale_end - noise_scale_start) * i / (num_steps - 1)
                for i in range(num_steps)
            ]
        else:
            noise_scale_schedule = [noise_scale_start]

        # 2. Three dictionaries
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width,
            "seed": seed,
            "noise_scale_start": noise_scale_start,
            "dtype": dtype,
            "device": device,
            "tokenizer": self.tokenizer,
            "processor": self.processor,
            "model_config": self.dit.config,
        }

        # 3. Unit chain
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        z = inputs_shared["latents"]

        # 4. Denoise loop (manual CFG in velocity space)
        self.load_models_to_device(self.in_iteration_models)

        for step_idx, step_t in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            sigma = (step_t / 1000.0).to(dtype=torch.float32).clamp_min(0.001)
            t_pixeldit = (1.0 - step_t / 1000.0).to(dtype=dtype, device=device)

            # Positive forward
            x_pred_cond = self.model_fn(
                dit=self.dit, latents=z, timestep=t_pixeldit,
                input_ids=inputs_posi["input_ids"],
                position_ids=inputs_posi["position_ids"],
                token_types=inputs_posi["token_types"],
                vinput_mask=inputs_posi["vinput_mask"],
            )
            v_cond = (x_pred_cond.to(torch.float32) - z.to(torch.float32)) / sigma

            # CFG
            if cfg_scale > 1.0:
                x_pred_uncond = self.model_fn(
                    dit=self.dit, latents=z, timestep=t_pixeldit,
                    input_ids=inputs_nega["input_ids"],
                    position_ids=inputs_nega["position_ids"],
                    token_types=inputs_nega["token_types"],
                    vinput_mask=inputs_nega["vinput_mask"],
                )
                v_uncond = (x_pred_uncond.to(torch.float32) - z.to(torch.float32)) / sigma
                v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_guided = v_cond

            model_output = -v_guided

            # Flash scheduler step
            z = _flash_step(
                self.scheduler.sigmas, step_idx, model_output, z,
                s_noise=noise_scale_schedule[step_idx],
                noise_clip_std=noise_clip_std,
            ).to(dtype)

        # 5. Decode
        img = (z + 1) / 2
        h_patches = height // 32
        w_patches = width // 32
        img = einops.rearrange(
            img.cpu().float(), 'B (H W) (C p1 p2) -> B C (H p1) (W p2)',
            H=h_patches, W=w_patches, p1=32, p2=32,
        )
        arr = np.round(np.clip(img[0].numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")


# ──────────────────────────────────────────────────────────────────────
# model_fn
# ──────────────────────────────────────────────────────────────────────
def model_fn_hidream_o1_image(
    dit, latents, timestep,
    input_ids, position_ids, token_types, vinput_mask,
    **kwargs
):
    outputs = dit(
        input_ids=input_ids,
        position_ids=position_ids,
        vinputs=latents,
        timestep=timestep.reshape(-1),
        token_types=token_types,
    )
    x_pred = outputs.x_pred[0, vinput_mask[0]].unsqueeze(0)
    return x_pred
