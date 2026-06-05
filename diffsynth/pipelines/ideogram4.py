from typing import Optional, Sequence, Union

import torch
from PIL import Image
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..core import ModelConfig
from ..models.ideogram4_dit import Ideogram4DiT, LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, IMAGE_POSITION_OFFSET
from ..models.ideogram4_text_encoder import Ideogram4TextEncoder
from ..models.ideogram4_vae import Ideogram4VAEEncoder, Ideogram4VAEDecoder
from transformers import AutoTokenizer


class Ideogram4Pipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("Ideogram4")
        self.text_encoder: Ideogram4TextEncoder = None
        self.dit: Ideogram4DiT = None
        self.dit_uncond: Ideogram4DiT = None
        self.vae_encoder: Ideogram4VAEEncoder = None
        self.vae_decoder: Ideogram4VAEDecoder = None
        self.tokenizer: AutoTokenizer = None
        self.in_iteration_models = ("dit", "dit_uncond")
        self.units = [
            Ideogram4Unit_ShapeChecker(),
            Ideogram4Unit_PromptEmbedder(),
            Ideogram4Unit_NoiseInitializer(),
            Ideogram4Unit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_ideogram4

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = Ideogram4Pipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        transformers = model_pool.fetch_model("ideogram4_dit", index=2)
        if isinstance(transformers, list):
            pipe.dit = transformers[0]
            pipe.dit_uncond = transformers[1]
        else:
            pipe.dit = transformers
        pipe.text_encoder = model_pool.fetch_model("ideogram4_text_encoder")
        pipe.vae_encoder = model_pool.fetch_model("ideogram4_vae_encoder")
        pipe.vae_decoder = model_pool.fetch_model("ideogram4_vae_decoder")

        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str = "",
        negative_prompt: str = "",
        cfg_scale: float = 7.0,
        # Input image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 50,
        # Progress bar
        progress_bar_cmd=tqdm,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, image_resolution=(height, width))

        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        self.load_models_to_device(self.in_iteration_models)
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)
            models = {"dit": self.dit}
            noise_pred_posi = self.model_fn(timestep=timestep, **models, **inputs_shared, **inputs_posi)
            if cfg_scale != 1:
                models = {"dit": self.dit_uncond if self.dit_uncond is not None else self.dit}
                noise_pred_nega = self.model_fn(timestep=timestep, **models, **inputs_shared, **inputs_nega)
                # This is not a standard CFG implementation. We align it to the original version of Ideogram4.
                noise_pred = cfg_scale * noise_pred_posi + (1.0 - cfg_scale) * noise_pred_nega
            else:
                noise_pred = noise_pred_posi

            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
            
        # Decode
        self.load_models_to_device(["vae_decoder"])
        image = self.vae_decoder.decode(inputs_shared["latents"], inputs_shared["grid_h"], inputs_shared["grid_w"], self.dit.patch_size, self.torch_dtype)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])
        return image


class Ideogram4Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: "Ideogram4Pipeline", height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class Ideogram4Unit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            output_params=("llm_features", "position_ids", "segment_ids", "indicator", "max_text_tokens"),
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: "Ideogram4Pipeline", inputs_shared, inputs_posi, inputs_nega):
        prompt = inputs_posi.get("prompt", "")
        height = inputs_shared.get("height")
        width = inputs_shared.get("width")
        max_text_tokens = 2048

        pipe.load_models_to_device(self.onload_model_names)

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = pipe.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        encoded = pipe.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = encoded["input_ids"][0]
        num_text_tokens = int(token_ids.shape[0])

        if num_text_tokens > max_text_tokens:
            raise ValueError(
                f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={max_text_tokens}"
            )

        patch = pipe.dit.patch_size * pipe.vae_encoder.ae_scale_factor
        grid_h = height // patch
        grid_w = width // patch
        num_image_tokens = grid_h * grid_w

        max_text_tokens = num_text_tokens
        total_seq_len = max_text_tokens + num_image_tokens

        h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        token_ids_padded = torch.zeros(1, total_seq_len, dtype=torch.long)
        text_position_ids = torch.zeros(1, total_seq_len, 3, dtype=torch.long)
        position_ids = torch.zeros(1, total_seq_len, 3, dtype=torch.long)
        segment_ids = torch.zeros(1, total_seq_len, dtype=torch.long)
        indicator = torch.zeros(1, total_seq_len, dtype=torch.long)

        token_ids_padded[0, :num_text_tokens] = token_ids
        text_pos = torch.arange(num_text_tokens)
        text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
        text_position_ids[0, :num_text_tokens] = text_pos_3d
        position_ids[0, :num_text_tokens] = text_pos_3d
        position_ids[0, num_text_tokens:] = image_pos

        indicator[0, :num_text_tokens] = LLM_TOKEN_INDICATOR
        indicator[0, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR
        segment_ids[0, :total_seq_len] = 1

        token_ids_padded = token_ids_padded.to(pipe.device)
        text_position_ids = text_position_ids.to(pipe.device)
        position_ids = position_ids.to(pipe.device)
        segment_ids = segment_ids.to(pipe.device)
        indicator = indicator.to(pipe.device)

        attention_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)
        pos_2d = text_position_ids[..., 0].contiguous()

        with torch.no_grad():
            llm_features = pipe.text_encoder(token_ids_padded, attention_mask, pos_2d)

        text_mask = attention_mask.to(llm_features.dtype).unsqueeze(-1)
        llm_features = llm_features * text_mask
        llm_features = llm_features.to(torch.float32)

        inputs_posi.update({
            "llm_features": llm_features,
            "position_ids": position_ids,
            "segment_ids": segment_ids,
            "indicator": indicator,
            "max_text_tokens": max_text_tokens,
        })
        inputs_nega.update({
            "llm_features": torch.zeros(1, num_image_tokens, llm_features.shape[-1], dtype=llm_features.dtype, device=llm_features.device),
            "position_ids": position_ids[:, max_text_tokens:],
            "segment_ids": segment_ids[:, max_text_tokens:],
            "indicator": indicator[:, max_text_tokens:],
            "max_text_tokens": 0,
        })
        return inputs_shared, inputs_posi, inputs_nega


class Ideogram4Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: "Ideogram4Pipeline", height, width, seed, rand_device):
        patch = pipe.dit.patch_size * pipe.vae_encoder.ae_scale_factor
        grid_h = height // patch
        grid_w = width // patch
        num_image_tokens = grid_h * grid_w
        latent_dim = pipe.dit.config.in_channels
        noise = pipe.generate_noise((1, num_image_tokens, latent_dim), seed=seed, rand_device=rand_device, rand_torch_dtype=torch.float32)
        return {"noise": noise, "grid_h": grid_h, "grid_w": grid_w}


class Ideogram4Unit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "height", "width", "grid_h", "grid_w"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: "Ideogram4Pipeline", input_image, noise, height, width, grid_h, grid_w):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(["vae_encoder"])
        image = pipe.preprocess_image(input_image)
        input_latents = pipe.vae_encoder.encode(image, grid_h, grid_w, pipe.dit.patch_size)

        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


def model_fn_ideogram4(
    dit: Ideogram4DiT = None,
    latents=None,
    timestep=None,
    llm_features=None,
    position_ids=None,
    segment_ids=None,
    indicator=None,
    max_text_tokens=0,
    **kwargs,
):
    t_ideogram4 = timestep.to(torch.float32)

    text_z_padding = torch.zeros(
        1, max_text_tokens, latents.shape[-1],
        dtype=torch.float32, device=latents.device,
    )
    z = torch.cat([text_z_padding, latents], dim=1)

    out = dit(
        llm_features=llm_features, x=z, t=t_ideogram4,
        position_ids=position_ids, segment_ids=segment_ids, indicator=indicator,
    )
    return -out[:, max_text_tokens:]
