import torch, math
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import numpy as np
from math import prod
from transformers import AutoTokenizer

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput
from ..utils.lora.merge import merge_lora

from ..models.anima_dit import AnimaDiT
from ..models.z_image_text_encoder import ZImageTextEncoder
from ..models.wan_video_vae import WanVideoVAE


class AnimaImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("Z-Image")
        self.text_encoder: ZImageTextEncoder = None
        self.dit: AnimaDiT = None
        self.vae: WanVideoVAE = None
        self.tokenizer: AutoTokenizer = None
        self.tokenizer_t5xxl: AutoTokenizer = None
        self.in_iteration_models = ("dit",)
        self.units = [
            AnimaUnit_ShapeChecker(),
            AnimaUnit_NoiseInitializer(),
            AnimaUnit_InputImageEmbedder(),
            AnimaUnit_PromptEmbedder(),
        ]
        self.model_fn = model_fn_anima
        self.compilable_models = ["dit"]
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen3-0.6B", origin_file_pattern="./"),
        tokenizer_t5xxl_config: ModelConfig = ModelConfig(model_id="stabilityai/stable-diffusion-3.5-large", origin_file_pattern="tokenizer_3/"),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = AnimaImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("z_image_text_encoder")
        pipe.dit = model_pool.fetch_model("anima_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        if tokenizer_t5xxl_config is not None:
            tokenizer_t5xxl_config.download_if_necessary()
            pipe.tokenizer_t5xxl = AutoTokenizer.from_pretrained(tokenizer_t5xxl_config.path)
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        sigma_shift: float = None,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
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
        image = self.vae.decode(inputs_shared["latents"].unsqueeze(2), device=self.device).squeeze(2)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class AnimaUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: AnimaImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}



class AnimaUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: AnimaImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}



class AnimaUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: AnimaImagePipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        if isinstance(input_image, list):
            input_latents = []
            for image in input_image:
                image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
                input_latents.append(pipe.vae.encode(image))
            input_latents = torch.concat(input_latents, dim=0)
        else:
            image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
            input_latents = pipe.vae.encode(image.unsqueeze(2), device=pipe.device).squeeze(2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


class AnimaUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_emb",),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(
        self,
        pipe: AnimaImagePipeline,
        prompt,
        device = None,
        max_sequence_length: int = 512,
    ):
        if isinstance(prompt, str):
            prompt = [prompt]

        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = pipe.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-1]
        
        t5xxl_text_inputs = pipe.tokenizer_t5xxl(
            prompt,
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        t5xxl_ids = t5xxl_text_inputs.input_ids.to(device)

        return prompt_embeds.to(pipe.torch_dtype), t5xxl_ids

    def process(self, pipe: AnimaImagePipeline, prompt):
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds, t5xxl_ids = self.encode_prompt(pipe, prompt, pipe.device)
        return {"prompt_emb": prompt_embeds, "t5xxl_ids": t5xxl_ids}


def model_fn_anima(
    dit: AnimaDiT = None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    t5xxl_ids=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs
):
    latents = latents.unsqueeze(2)
    timestep = timestep / 1000
    model_output = dit(
        x=latents,
        timesteps=timestep,
        context=prompt_emb,
        t5xxl_ids=t5xxl_ids,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    model_output = model_output.squeeze(2)
    return model_output
