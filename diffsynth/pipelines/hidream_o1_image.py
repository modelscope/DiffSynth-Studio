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
        self.processor = None

        self.in_iteration_models = ("dit",)
        self.units = [
            HiDreamO1ImageUnit_ShapeChecker(),
            HiDreamO1ImageUnit_PromptTokenizer(),
            HiDreamO1ImageUnit_NoiseInitializer(),
        ]
        self.model_fn = model_fn_hidream_o1_image

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = ModelConfig(model_id="HiDream-ai/HiDream-O1-Image-Dev", origin_file_pattern="./"),
        vram_limit: float = None,
    ):
        pipe = HiDreamO1ImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.dit = model_pool.fetch_model("hidream_o1_image_dit")
        if processor_config is not None:
            from transformers import AutoProcessor
            processor_config.download_if_necessary()
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
            add_special_tokens(pipe.processor.tokenizer)
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
        # 1. Scheduler: set timesteps for Dev mode
        self.scheduler.set_timesteps(
            num_inference_steps, shift=shift,
            special_case="flash"
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
        }

        # 3. Unit chain
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # 4. Denoise loop (manual CFG in velocity space)
        self.load_models_to_device(self.in_iteration_models)

        for step_idx, step_t in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            sigma = (step_t / 1000.0).to(dtype=self.torch_dtype).clamp_min(0.001)
            t_pixeldit = (1.0 - step_t / 1000.0).to(dtype=self.torch_dtype, device=self.device)

            # Positive forward
            x_pred_cond = self.model_fn(
                dit=self.dit, latents=inputs_shared["latents"], timestep=t_pixeldit,
                input_ids=inputs_posi["input_ids"],
                position_ids=inputs_posi["position_ids"],
                token_types=inputs_posi["token_types"],
                vinput_mask=inputs_posi["vinput_mask"],
            )
            v_cond = (x_pred_cond.to(torch.float32) - inputs_shared["latents"].to(torch.float32)) / sigma

            # CFG
            if cfg_scale > 1.0:
                x_pred_uncond = self.model_fn(
                    dit=self.dit, latents=inputs_shared["latents"], timestep=t_pixeldit,
                    input_ids=inputs_nega["input_ids"],
                    position_ids=inputs_nega["position_ids"],
                    token_types=inputs_nega["token_types"],
                    vinput_mask=inputs_nega["vinput_mask"],
                )
                v_uncond = (x_pred_uncond.to(torch.float32) - inputs_shared["latents"].to(torch.float32)) / sigma
                v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_guided = v_cond

            model_output = -v_guided

            # Flash scheduler step
            inputs_shared["latents"] = _flash_step(
                self.scheduler.sigmas, step_idx, model_output, inputs_shared["latents"],
                s_noise=noise_scale_schedule[step_idx],
                noise_clip_std=noise_clip_std,
            ).to(self.torch_dtype)

        # 5. Decode
        img = (inputs_shared["latents"] + 1) / 2
        h_patches = height // 32
        w_patches = width // 32
        img = einops.rearrange(
            img.cpu().float(), 'B (H W) (C p1 p2) -> B C (H p1) (W p2)',
            H=h_patches, W=w_patches, p1=32, p2=32,
        )
        arr = np.round(np.clip(img[0].numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

class HiDreamO1ImageUnit_NoiseInitializer(PipelineUnit):
    """Generate pixel-level noise and rearrange to patch space."""

    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "noise_scale_start"),
            output_params=("latents",),
        )

    def prepare_inputs(self, inputs_shared, inputs_posi, inputs_nega):
        return inputs_shared, inputs_posi, inputs_nega

    def process(self, pipe: HiDreamO1ImagePipeline, height=None, width=None, seed=None, rand_device=None, noise_scale_start=None):
        noise = pipe.generate_noise((1, 3, height, width), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        noise = noise_scale_start * noise
        latents = einops.rearrange(
            noise, 'B C (H p1) (W p2) -> B (H W) (C p1 p2)',
            p1=pipe.height_division_factor, p2=pipe.width_division_factor,
        )
        return {"latents": latents}

class HiDreamO1ImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: HiDreamO1ImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class HiDreamO1ImageUnit_PromptTokenizer(PipelineUnit):
    """Tokenize prompt into input_ids, position_ids, token_types, vinput_mask."""

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=("height", "width"),
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("input_ids", "position_ids", "token_types", "vinput_mask"),
        )

    def process(self, pipe: HiDreamO1ImagePipeline, prompt, height, width):
        result = self.build_text_sample(
            prompt=prompt,
            height=height, width=width,
            tokenizer=pipe.processor.tokenizer, processor=pipe.processor,
            model_config=pipe.dit.config, device=pipe.device,
        )
        return result

    def build_text_sample(self, prompt, height, width, tokenizer, processor, model_config, device):
        TIMESTEP_TOKEN_NUM = 1
        PATCH_SIZE = 32
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

        position_ids, _ = get_rope_index_fix_point(
            spatial_merge_size=1,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            input_ids=input_ids_pad,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=None,
            skip_vision_start_token=[1],
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
