import torch
from typing import Union
from tqdm import tqdm
from transformers import AutoTokenizer

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.moss_image_text_encoder import MOSSImageTextEncoder
from ..models.moss_image_dit import MOSSImageDiT
from ..models.flux2_vae import Flux2VAE


class MOSSImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("MOSS-Image")
        self.text_encoder: MOSSImageTextEncoder = None
        self.dit: MOSSImageDiT = None
        self.vae: Flux2VAE = None
        self.tokenizer: AutoTokenizer = None

        self.in_iteration_models = ("dit",)
        self.units = [
            MOSSImageUnit_ShapeChecker(),
            MOSSImageUnit_PromptEmbedder(),
            MOSSImageUnit_NoiseInitializer(),
            MOSSImageUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_moss_image
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
    ):
        pipe = MOSSImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("moss_image_text_encoder")
        pipe.dit = model_pool.fetch_model("moss_image_dit")
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
        cfg_scale: float = 3.0,
        max_sequence_length: int = 4096,
        uncond_skip_layers: tuple[int] = (22,),
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        # Progress bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, shift=sigma_shift)

        # Parameters
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt, "skip_layers": uncond_skip_layers}
        inputs_shared = {
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "cfg_scale": cfg_scale, "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)
            # Keep the sampling state in fp32; only the model input is cast to the compute dtype.
            model_inputs = {**inputs_shared, "latents": inputs_shared["latents"].to(self.torch_dtype)}
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                model_inputs, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

        # Decode
        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"]
        image = self.vae.decode(latents)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])
        return image


class MOSSImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: MOSSImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class MOSSImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("max_sequence_length",),
            output_params=("prompt_embeds", "prompt_embeds_mask"),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(self, pipe: MOSSImagePipeline, prompt, max_sequence_length):
        formatted = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        inputs = pipe.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_sequence_length)
        inputs = {name: value.to(pipe.device) for name, value in inputs.items()}
        prompt_embeds = pipe.text_encoder(**inputs, use_cache=False).to(pipe.torch_dtype)
        prompt_embeds_mask = inputs["attention_mask"].bool()
        return {"prompt_embeds": prompt_embeds, "prompt_embeds_mask": prompt_embeds_mask}

    def process(self, pipe: MOSSImagePipeline, prompt, max_sequence_length):
        pipe.load_models_to_device(self.onload_model_names)
        if pipe.text_encoder is None or pipe.tokenizer is None:
            return {}
        return self.encode_prompt(pipe, prompt, max_sequence_length)


class MOSSImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: MOSSImagePipeline, height, width, seed, rand_device):
        latent_h = height // pipe.height_division_factor
        latent_w = width // pipe.width_division_factor
        latent_channels = pipe.dit.latent_channels
        if rand_device is None:
            rand_device = str(pipe.device)
        noise = pipe.generate_noise(
            (1, latent_channels, latent_h, latent_w),
            seed=seed, rand_device=rand_device,
            rand_torch_dtype=torch.float32, torch_dtype=torch.float32,
        )
        return {"noise": noise}


class MOSSImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: MOSSImagePipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}

        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae.encode(image)

        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


def model_fn_moss_image(
    dit: MOSSImageDiT,
    latents=None,
    timestep=None,
    prompt_embeds=None,
    prompt_embeds_mask=None,
    skip_layers=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    output = dit(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_embeds_mask,
        skip_layers=skip_layers,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return output.float()
