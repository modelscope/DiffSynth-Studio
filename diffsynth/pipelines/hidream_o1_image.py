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
from ..models.hidream_common import add_special_tokens, get_rope_index_fix_point, patchify, unpatchify, PATCH_SIZE
from transformers import AutoTokenizer


# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────

class HiDreamO1ImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=PATCH_SIZE, width_division_factor=PATCH_SIZE,
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
        cfg_scale: float = 4.0,
        height: int = 2048,
        width: int = 2048,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 50,
        model_type: str = "full",
        shift: float = 3.0,
        noise_scale: float = 8.0,
        progress_bar_cmd=tqdm,
    ):
        # 1. Scheduler: set timesteps for Dev mode
        self.scheduler.set_timesteps(
            num_inference_steps, shift=shift,
            special_case=model_type,
        )

        # 2. Input Dictionaries
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "noise_scale": noise_scale,
        }

        # 3. Units
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # 4. Denoise loop
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

        image = self.vae_output_to_image(inputs_shared["latents"])
        return image

class HiDreamO1ImageUnit_NoiseInitializer(PipelineUnit):
    """Generate pixel-level noise and rearrange to patch space."""

    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "noise_scale"),
            output_params=("latents",),
        )

    def prepare_inputs(self, inputs_shared, inputs_posi, inputs_nega):
        return inputs_shared, inputs_posi, inputs_nega

    def process(self, pipe: HiDreamO1ImagePipeline, height=None, width=None, seed=None, rand_device=None, noise_scale=None):
        noise = pipe.generate_noise((1, 3, height, width), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        noise = noise_scale * noise
        return {"latents": noise}


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

    b, c, h, w = latents.shape
    x = patchify(latents)

    timestep = timestep / 1000
    outputs = dit(
        input_ids=input_ids,
        position_ids=position_ids,
        vinputs=x,
        timestep=(1 - timestep).reshape(-1),
        token_types=token_types,
    )
    x_pred = unpatchify(outputs.x_pred[0, vinput_mask[0]].unsqueeze(0), h, w)
    v_pred = (latents - x_pred) / timestep
    return v_pred
