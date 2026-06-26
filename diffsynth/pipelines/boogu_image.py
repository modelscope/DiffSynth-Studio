import torch, math
import numpy as np
from PIL import Image
from typing import List, Optional, Union
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..diffusion.flow_match import FlowMatchScheduler
from ..models.boogu_image_dit import BooguImageDiT, BooguImageDoubleStreamRotaryPosEmbed
from ..models.joyai_image_text_encoder import JoyAIImageTextEncoder
from ..models.flux_vae import FluxVAEEncoder, FluxVAEDecoder


def _resize_image_boogu(image, max_pixels=None, max_side_length=None, scale_factor=16):
    h, w = image.height, image.width
    ratio = 1.0
    if max_side_length is not None:
        ratio = min(ratio, max_side_length / max(h, w))
    if max_pixels is not None:
        ratio = min(ratio, (max_pixels / (h * w)) ** 0.5)
    ratio = min(ratio, 1.0)
    new_h = int(h * ratio) // scale_factor * scale_factor
    new_w = int(w * ratio) // scale_factor * scale_factor
    return image.resize((new_w, new_h))


class BooguImagePipeline(BasePipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=16, width_division_factor=16)
        self.scheduler = FlowMatchScheduler("Boogu")
        self.text_encoder: JoyAIImageTextEncoder = None
        self.dit: BooguImageDiT = None
        self.vae_encoder: FluxVAEEncoder = None
        self.vae_decoder: FluxVAEDecoder = None
        self.processor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            BooguImageUnit_ShapeChecker(),
            BooguImageUnit_PromptEmbedder(),
            BooguImageUnit_NoiseInitializer(),
            BooguImageUnit_InputImageEmbedder(),
            BooguImageUnit_EditImageEmbedder(),
            BooguImageUnit_FreqsCis(),
        ]
        self.model_fn = model_fn_boogu_image
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = BooguImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("joyai_image_text_encoder")
        pipe.dit = model_pool.fetch_model("boogu_image_dit")
        pipe.vae_encoder = model_pool.fetch_model("flux_vae_encoder")
        pipe.vae_decoder = model_pool.fetch_model("flux_vae_decoder")
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import AutoProcessor
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        input_image: Image.Image = None,
        edit_image: Image.Image = None,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        denoising_strength: float = 1.0,
        sigmas: list[float] = None,
        num_inference_steps: int = 20,
        max_sequence_length: int = 1280,
        max_input_image_pixels: int = 4194304,
        max_input_image_side_length: int = 4096,
        max_vlm_input_pil_pixels: int = 147456,
        max_vlm_input_pil_side_length: int = 768,
        rand_device: str = "cpu",
        progress_bar_cmd=tqdm,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, sigmas=sigmas)
        
        # Parameters
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image,
            "edit_image": edit_image,
            "height": height, "width": width,
            "seed": seed, "max_sequence_length": max_sequence_length,
            "max_input_image_pixels": max_input_image_pixels,
            "max_input_image_side_length": max_input_image_side_length,
            "max_vlm_input_pil_pixels": max_vlm_input_pil_pixels,
            "max_vlm_input_pil_side_length": max_vlm_input_pil_side_length,
            "rand_device": rand_device,
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
        self.load_models_to_device(["vae_decoder"])
        latents = inputs_shared["latents"]
        image = self.vae_decoder(latents, device=self.device)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])
        return image


class BooguImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width"), output_params=("height", "width"))

    def process(self, pipe: BooguImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


SYSTEM_PROMPT_TI2I = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."


class BooguImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=(
                "max_sequence_length",
                "edit_image",
                "max_vlm_input_pil_pixels",
                "max_vlm_input_pil_side_length",
            ),
            onload_model_names=("text_encoder",),
        )

    def encode_prompt(self, pipe: BooguImagePipeline, prompt, max_sequence_length, edit_image=None, max_vlm_input_pil_pixels=147456, max_vlm_input_pil_side_length=768):
        if edit_image is not None:
            system_prompt = SYSTEM_PROMPT_TI2I
            vlm_image = _resize_image_boogu(edit_image, max_pixels=max_vlm_input_pil_pixels, max_side_length=max_vlm_input_pil_side_length)
            user_content = [{"type": "image", "image": vlm_image}, {"type": "text", "text": prompt}]
        else:
            system_prompt = "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows."
            user_content = [{"type": "text", "text": prompt}]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        vlm_inputs = pipe.processor.apply_chat_template(
            [messages],
            padding="longest",
            max_length=max_sequence_length,
            padding_side="right",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        vlm_inputs = {k: v.to(pipe.device) if isinstance(v, torch.Tensor) else v for k, v in vlm_inputs.items()}
        instruction_hidden_states = pipe.text_encoder(**vlm_inputs)
        instruction_hidden_states = instruction_hidden_states.to(dtype=pipe.torch_dtype, device=pipe.device)
        instruction_attention_mask = vlm_inputs["attention_mask"].to(device=pipe.device)
        return instruction_hidden_states, instruction_attention_mask

    def process(self, pipe: BooguImagePipeline, prompt, max_sequence_length, edit_image=None, max_vlm_input_pil_pixels=147456, max_vlm_input_pil_side_length=768):
        pipe.load_models_to_device(["text_encoder"])
        instruction_hidden_states, instruction_attention_mask = self.encode_prompt(pipe, prompt, max_sequence_length, edit_image, max_vlm_input_pil_pixels, max_vlm_input_pil_side_length)
        return {"instruction_hidden_states": instruction_hidden_states, "instruction_attention_mask": instruction_attention_mask}


class BooguImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: BooguImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise(
            (1, 16, height // 8, width // 8),
            seed=seed, rand_device=rand_device, rand_torch_dtype=torch.float32,
            device=pipe.device, torch_dtype=pipe.torch_dtype,
        )
        return {"noise": noise}


class BooguImageUnit_FreqsCis(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("freqs_cis",),
        )

    def process(self, pipe: BooguImagePipeline, height, width):
        axes_dim_rope = (40, 40, 40)
        axes_lens = (2048, 1664, 1664)
        freqs_cis = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(
            axes_dim_rope, axes_lens, theta=10000,
        )
        return {"freqs_cis": freqs_cis}


class BooguImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: BooguImagePipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(["vae_encoder"])
        image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": None}


class BooguImageUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "max_input_image_pixels", "max_input_image_side_length"),
            output_params=("ref_image_hidden_states",),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: BooguImagePipeline, edit_image, max_input_image_pixels, max_input_image_side_length):
        if edit_image is None:
            return {"ref_image_hidden_states": None}
        pipe.load_models_to_device(["vae_encoder"])
        image = _resize_image_boogu(edit_image, max_pixels=max_input_image_pixels, max_side_length=max_input_image_side_length)
        image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
        ref_latents = pipe.vae_encoder(image).squeeze(0)
        return {"ref_image_hidden_states": [[ref_latents]]}


def model_fn_boogu_image(
    dit,
    latents,
    timestep,
    instruction_hidden_states,
    freqs_cis,
    instruction_attention_mask,
    ref_image_hidden_states=None,
    **kwargs,
):
    output = dit(
        hidden_states=latents,
        timestep=timestep,
        instruction_hidden_states=instruction_hidden_states,
        freqs_cis=freqs_cis,
        instruction_attention_mask=instruction_attention_mask,
        ref_image_hidden_states=ref_image_hidden_states,
    )
    return -output
