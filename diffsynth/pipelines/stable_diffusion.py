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
from ..models.stable_diffusion_unet import UNet2DConditionModel
from ..models.stable_diffusion_vae import StableDiffusionVAE


class StableDiffusionPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.float16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=8, width_division_factor=8,
        )
        self.scheduler = DDIMScheduler()
        self.text_encoder: SDTextEncoder = None
        self.unet: UNet2DConditionModel = None
        self.vae: StableDiffusionVAE = None
        self.tokenizer: AutoTokenizer = None

        self.in_iteration_models = ("unet",)
        self.units = [
            SDUnit_ShapeChecker(),
            SDUnit_PromptEmbedder(),
            SDUnit_NoiseInitializer(),
            SDUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_stable_diffusion
        self.compilable_models = ["unet"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = StableDiffusionPipeline(device=device, torch_dtype=torch_dtype)
        # Override vram_config to use the specified torch_dtype for all models
        for mc in model_configs:
            mc._vram_config_override = {
                'onload_dtype': torch_dtype,
                'computation_dtype': torch_dtype,
            }
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("stable_diffusion_text_encoder")
        pipe.unet = model_pool.fetch_model("stable_diffusion_unet")
        pipe.vae = model_pool.fetch_model("stable_diffusion_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 50,
        eta: float = 0.0,
        guidance_rescale: float = 0.0,
        progress_bar_cmd=tqdm,
    ):
        # 1. Scheduler
        self.scheduler.set_timesteps(
            num_inference_steps, eta=eta,
        )

        # 2. Three-dict input preparation
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "guidance_rescale": guidance_rescale,
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
            inputs_shared["latents"] = self.step(
                self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared
            )

        # 5. VAE decode
        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"] / self.vae.scaling_factor
        image = self.vae.decode(latents)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class SDUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: StableDiffusionPipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class SDUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_embeds",),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(
        self,
        pipe: StableDiffusionPipeline,
        prompt: str,
        device: torch.device,
    ) -> torch.Tensor:
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = pipe.text_encoder(text_input_ids)
        # TextEncoder returns (last_hidden_state, hidden_states) or just last_hidden_state.
        # last_hidden_state is the post-final-layer-norm output, matching diffusers encode_prompt.
        if isinstance(prompt_embeds, tuple):
            prompt_embeds = prompt_embeds[0]
        return prompt_embeds

    def process(self, pipe: StableDiffusionPipeline, prompt):
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds = self.encode_prompt(pipe, prompt, pipe.device)
        return {"prompt_embeds": prompt_embeds}


class SDUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: StableDiffusionPipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype
        )
        return {"noise": noise}


class SDUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: StableDiffusionPipeline, input_image, noise):
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


def model_fn_stable_diffusion(
    unet: UNet2DConditionModel,
    latents=None,
    timestep=None,
    prompt_embeds=None,
    cross_attention_kwargs=None,
    timestep_cond=None,
    added_cond_kwargs=None,
    **kwargs,
):
    # SD timestep is already in 0-999 range, no scaling needed
    noise_pred = unet(
        latents,
        timestep,
        encoder_hidden_states=prompt_embeds,
        cross_attention_kwargs=cross_attention_kwargs,
        timestep_cond=timestep_cond,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
    return noise_pred
