import torch
from PIL import Image
from tqdm import tqdm
from typing import Union

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion.ddim_scheduler import DDIMScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from transformers import AutoTokenizer, CLIPTextModel
from ..models.stable_diffusion_text_encoder import SDTextEncoder
from ..models.stable_diffusion_xl_unet import SDXLUNet2DConditionModel
from ..models.stable_diffusion_xl_text_encoder import SDXLTextEncoder2
from ..models.stable_diffusion_vae import StableDiffusionVAE


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """Rescale noise_cfg based on guidance_rescale to prevent overexposure.

    Based on Section 3.4 from "Common Diffusion Noise Schedules and Sample Steps are Flawed"
    https://huggingface.co/papers/2305.08891
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class StableDiffusionXLPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=8, width_division_factor=8,
        )
        self.scheduler = DDIMScheduler()
        self.text_encoder: SDTextEncoder = None
        self.text_encoder_2: SDXLTextEncoder2 = None
        self.unet: SDXLUNet2DConditionModel = None
        self.vae: StableDiffusionVAE = None
        self.tokenizer: AutoTokenizer = None
        self.tokenizer_2: AutoTokenizer = None

        self.in_iteration_models = ("unet",)
        self.units = [
            SDXLUnit_ShapeChecker(),
            SDXLUnit_PromptEmbedder(),
            SDXLUnit_NoiseInitializer(),
            SDXLUnit_InputImageEmbedder(),
            SDXLUnit_AddTimeIdsComputer(),
        ]
        self.model_fn = model_fn_stable_diffusion_xl
        self.compilable_models = ["unet"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        tokenizer_2_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = StableDiffusionXLPipeline(device=device, torch_dtype=torch_dtype)
        # Override vram_config to use the specified torch_dtype for all models
        for mc in model_configs:
            mc._vram_config_override = {
                'onload_dtype': torch_dtype,
                'computation_dtype': torch_dtype,
            }
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("stable_diffusion_text_encoder")
        pipe.text_encoder_2 = model_pool.fetch_model("stable_diffusion_xl_text_encoder")
        pipe.unet = model_pool.fetch_model("stable_diffusion_xl_unet")
        pipe.vae = model_pool.fetch_model("stable_diffusion_xl_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        if tokenizer_2_config is not None:
            tokenizer_2_config.download_if_necessary()
            pipe.tokenizer_2 = AutoTokenizer.from_pretrained(tokenizer_2_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 5.0,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 50,
        guidance_rescale: float = 0.0,
        progress_bar_cmd=tqdm,
    ):
        # 1. Scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # 2. Three-dict input preparation
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "guidance_rescale": guidance_rescale,
            "crops_coords_top_left": (0, 0),
        }

        # 3. Unit chain execution
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

            # Apply guidance_rescale
            if guidance_rescale > 0.0:
                # cfg_guided_model_fn already applied CFG, now apply rescale
                # We need the text-only prediction for rescale
                noise_pred_text = self.model_fn(
                    self.unet,
                    inputs_shared["latents"],
                    timestep,
                    inputs_posi["prompt_embeds"],
                    pooled_prompt_embeds=inputs_posi["pooled_prompt_embeds"],
                    add_time_ids=inputs_posi["add_time_ids"],
                )
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

            inputs_shared["latents"] = self.step(
                self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared
            )

        # 6. VAE decode
        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"] / self.vae.scaling_factor
        image = self.vae.decode(latents)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class SDXLUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: StableDiffusionXLPipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class SDXLUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "prompt"},
            output_params=("prompt_embeds", "pooled_prompt_embeds"),
            onload_model_names=("text_encoder", "text_encoder_2")
        )

    def encode_prompt(
        self,
        pipe: StableDiffusionXLPipeline,
        prompt: str,
        device: torch.device,
    ) -> tuple:
        """Encode prompt using both text encoders (same prompt for both).

        Returns (prompt_embeds, pooled_prompt_embeds):
          - prompt_embeds: concat(encoder1_output, encoder2_output) -> (B, 77, 2048)
          - pooled_prompt_embeds: encoder2 pooled output -> (B, 1280)
        """
        # Text Encoder 1 (CLIP-L, 768-dim)
        text_input_ids_1 = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        prompt_embeds_1 = pipe.text_encoder(text_input_ids_1)
        if isinstance(prompt_embeds_1, tuple):
            prompt_embeds_1 = prompt_embeds_1[0]

        # Text Encoder 2 (CLIP-bigG, 1280-dim) — uses penultimate hidden states + pooled
        text_input_ids_2 = pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        # SDXLTextEncoder2 forward returns (text_embeds/pooled, hidden_states_tuple)
        pooled_prompt_embeds, hidden_states = pipe.text_encoder_2(text_input_ids_2, output_hidden_states=True)
        # Use penultimate hidden state (same as diffusers: hidden_states[-2])
        prompt_embeds_2 = hidden_states[-2]

        # Concatenate both encoder outputs along feature dimension
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        return prompt_embeds, pooled_prompt_embeds

    def process(self, pipe: StableDiffusionXLPipeline, prompt):
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(pipe, prompt, pipe.device)
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}


class SDXLUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: StableDiffusionXLPipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype
        )
        return {"noise": noise}


class SDXLUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: StableDiffusionXLPipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_tensor = pipe.preprocess_image(input_image)
        input_latents = pipe.vae.encode(input_tensor).sample() * pipe.vae.scaling_factor
        latents = pipe.scheduler.add_noise(input_latents, noise, pipe.scheduler.timesteps[0])
        if pipe.scheduler.training:
            return {"latents": latents, "input_latents": input_latents}
        else:
            return {"latents": latents}


class SDXLUnit_AddTimeIdsComputer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("add_time_ids",),
        )

    def _get_add_time_ids(self, pipe, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        expected_add_embed_dim = pipe.unet.add_embedding.linear_1.in_features
        addition_time_embed_dim = pipe.unet.add_time_proj.num_channels
        passed_add_embed_dim = addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
                f"but a vector of {passed_add_embed_dim} was created."
            )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=pipe.device)
        return add_time_ids

    def process(self, pipe: StableDiffusionXLPipeline, height, width):
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
        add_time_ids = self._get_add_time_ids(
            pipe, original_size, crops_coords_top_left, target_size,
            dtype=pipe.torch_dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        return {"add_time_ids": add_time_ids}


def model_fn_stable_diffusion_xl(
    unet: SDXLUNet2DConditionModel,
    latents=None,
    timestep=None,
    prompt_embeds=None,
    pooled_prompt_embeds=None,
    add_time_ids=None,
    cross_attention_kwargs=None,
    timestep_cond=None,
    **kwargs,
):
    """SDXL model forward with added_cond_kwargs for micro-conditioning."""
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids,
    }
    noise_pred = unet(
        latents,
        timestep,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        cross_attention_kwargs=cross_attention_kwargs,
        timestep_cond=timestep_cond,
        return_dict=False,
    )[0]
    return noise_pred
