"""
ERNIE-Image Text-to-Image Pipeline for DiffSynth-Studio.

Architecture: SharedAdaLN DiT + RoPE 3D + Joint Image-Text Attention.
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, List, Optional

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from tqdm import tqdm
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from transformers import AutoTokenizer
from ..models.ernie_image_text_encoder import ErnieImageTextEncoder
from ..models.ernie_image_dit import ErnieImageDiT
from ..models.flux2_vae import Flux2VAE


# ============================================================
# Pipeline
# ============================================================

class ErnieImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("ERNIE-Image")
        self.text_encoder: ErnieImageTextEncoder = None
        self.dit: ErnieImageDiT = None
        self.vae: Flux2VAE = None
        self.tokenizer: AutoTokenizer = None
        self.in_iteration_models = ("dit",)
        self.units = [
            ErnieImageUnit_ShapeChecker(),
            ErnieImageUnit_PromptEmbedder(),
            ErnieImageUnit_NoiseInitializer(),
        ]
        self.model_fn = model_fn_ernie_image
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
    ):
        pipe = ErnieImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("ernie_image_text_encoder")
        pipe.dit = model_pool.fetch_model("ernie_image_dit")
        pipe.vae = model_pool.fetch_model("flux2_vae")

        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        initial_noise: torch.Tensor = None,
        # Steps
        num_inference_steps: int = 50,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        # Scheduler: FLUX.2 template with exponential shift sigma schedule
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device, "initial_noise": initial_noise,
            "num_inference_steps": num_inference_steps,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
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

        # Decode
        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"]
        # VAE decode (handles BN unnormalization and unpatchify internally)
        image = self.vae.decode(latents)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


# ============================================================
# PipelineUnits
# ============================================================

class ErnieImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: ErnieImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class ErnieImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_embeds", "prompt_embeds_mask"),
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: ErnieImagePipeline, prompt):
        if isinstance(prompt, str):
            prompt = [prompt]

        # Get text embeddings from encoder
        text_hiddens = []
        text_lens_list = []
        for p in prompt:
            ids = pipe.tokenizer(
                p,
                add_special_tokens=True,
                truncation=True,
                padding=False,
            )["input_ids"]

            if len(ids) == 0:
                if pipe.tokenizer.bos_token_id is not None:
                    ids = [pipe.tokenizer.bos_token_id]
                else:
                    ids = [0]

            input_ids = torch.tensor([ids], device=pipe.device)
            outputs = pipe.text_encoder(
                input_ids=input_ids,
            )
            # Text encoder returns tuple of (hidden_states_tuple,)
            # where hidden_states_tuple is a tuple of hidden states from each layer
            all_hidden_states = outputs[0]  # tuple of hidden states
            hidden = all_hidden_states[-2][0]  # [T, H] - second to last layer
            text_hiddens.append(hidden)
            text_lens_list.append(hidden.shape[0])

        # Pad to uniform length
        if len(text_hiddens) == 0:
            text_in_dim = pipe.text_encoder.config.hidden_size if hasattr(pipe.text_encoder, 'config') else 3072
            return {
                "prompt_embeds": torch.zeros((0, 0, text_in_dim), device=pipe.device, dtype=pipe.torch_dtype),
                "prompt_embeds_mask": torch.zeros((0,), device=pipe.device, dtype=torch.long),
            }

        normalized = [th.to(pipe.device).to(pipe.torch_dtype) for th in text_hiddens]
        text_lens = torch.tensor([t.shape[0] for t in normalized], device=pipe.device, dtype=torch.long)
        Tmax = int(text_lens.max().item())
        text_in_dim = normalized[0].shape[1]
        text_bth = torch.zeros((len(normalized), Tmax, text_in_dim), device=pipe.device, dtype=pipe.torch_dtype)
        for i, t in enumerate(normalized):
            text_bth[i, :t.shape[0], :] = t

        return {"prompt_embeds": text_bth, "prompt_embeds_mask": text_lens}


class ErnieImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "initial_noise"),
            output_params=("latents",),
        )

    def process(self, pipe: ErnieImagePipeline, height, width, seed, rand_device, initial_noise):
        if initial_noise is not None:
            return {"latents": initial_noise}

        latent_h = height // 16
        latent_w = width // 16
        latent_channels = pipe.dit.in_channels

        # Use pipeline device if rand_device is not specified or is "cpu" with CUDA pipeline
        if rand_device is None or (rand_device == "cpu" and "cuda" in str(pipe.device)):
            rand_device = str(pipe.device)

        generator = torch.Generator(device=rand_device)
        if seed is not None:
            generator.manual_seed(seed)

        noise = torch.randn(
            (1, latent_channels, latent_h, latent_w),
            device=pipe.device,
            dtype=pipe.torch_dtype,
            generator=generator,
        )
        return {"latents": noise}


# ============================================================
# model_fn
# ============================================================

def model_fn_ernie_image(
    dit: ErnieImageDiT,
    latents=None,
    timestep=None,
    prompt_embeds=None,
    prompt_embeds_mask=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    output = dit(
        hidden_states=latents,
        timestep=timestep,
        text_bth=prompt_embeds,
        text_lens=prompt_embeds_mask,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return output
