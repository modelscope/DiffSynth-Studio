import math
from typing import Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageOps
from tqdm import tqdm

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.flux2_vae import Flux2VAE
from ..models.flux3_dit import IN_CHANNELS, MODES, Flux3DiT, Packed
from ..models.krea2_text_encoder import Krea2TextEncoder


class Flux3ImagePipeline(BasePipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=16, width_division_factor=16)
        self.scheduler = FlowMatchScheduler("FLUX.3")
        self.text_encoder: Krea2TextEncoder = None
        self.dit: Flux3DiT = None
        self.vae: Flux2VAE = None
        self.processor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            Flux3Unit_ShapeChecker(),
            Flux3Unit_PromptEmbedder(),
            Flux3Unit_NoiseInitializer(),
            Flux3Unit_InputImageEmbedder(),
            Flux3Unit_ImageIDs(),
            Flux3Unit_ReferenceEmbedder(),
            Flux3Unit_PackedBuilder(),
        ]
        self.model_fn = model_fn_flux3
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/"),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = Flux3ImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("krea2_text_encoder")
        pipe.dit = model_pool.fetch_model("flux3_dit")
        pipe.vae = model_pool.fetch_model("flux2_vae")
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import AutoProcessor
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str = "",
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Edit
        edit_image: Union[Image.Image, list[Image.Image]] = None,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 16,
        alpha: float = 6.93,
        # Progress bar
        progress_bar_cmd=tqdm,
    ):
        if self.dit.flow_map and cfg_scale != 1.0:
            raise ValueError("FLUX.3 Image Klein Distilled samples without CFG; set cfg_scale=1.0")
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, alpha=alpha)

        # Parameters
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale, "input_image": input_image, "edit_image": edit_image,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.reshape(1).to(dtype=torch.float32, device=self.device)
            sigmas = torch.cat([self.scheduler.sigmas, self.scheduler.sigmas.new_zeros(1)])
            timesteps_ref = sigmas[progress_id + 1].reshape(1).to(self.device) if self.dit.flow_map else None
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, timesteps_ref=timesteps_ref
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

        # Decode
        self.load_models_to_device(("vae",))
        latents = rearrange(inputs_shared["latents"], "(H W) C -> 1 C H W", H=inputs_shared["height"] // 16, W=inputs_shared["width"] // 16)
        image = self.vae.decode(latents)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class Flux3Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class Flux3Unit_PromptEmbedder(PipelineUnit):
    _OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
    _MAX_LENGTH = 4096
    _PAD_MULTIPLE = 80

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_tokens", "prompt_ids"),
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe, prompt):
        if pipe.text_encoder is None or pipe.processor is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        text = pipe.processor.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        tokenizer = pipe.processor.tokenizer
        real_length = tokenizer(text, return_tensors="pt", truncation=True, max_length=self._MAX_LENGTH)["input_ids"].shape[1]
        length = min(math.ceil(real_length / self._PAD_MULTIPLE) * self._PAD_MULTIPLE, self._MAX_LENGTH)
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        mask = inputs["attention_mask"].to(pipe.device)
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        hidden_states = pipe.text_encoder(
            input_ids=inputs["input_ids"].to(pipe.device), attention_mask=mask,
            position_ids=positions[None].expand(4, -1, -1), use_cache=False,
        )
        stacked = torch.stack([hidden_states[index] for index in self._OUTPUT_LAYERS], dim=1)
        tokens = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(pipe.torch_dtype)
        positions = torch.arange(length, device=pipe.device)
        ids = torch.stack([torch.zeros_like(positions)] * 3 + [positions], dim=-1)
        return {"prompt_tokens": tokens, "prompt_ids": ids}


class Flux3Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe, height, width, seed, rand_device):
        shape = (height // 16 * width // 16, IN_CHANNELS["image"])
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}


class Flux3Unit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image)
        latent = pipe.vae.encode(image)
        input_latents = latent[0].permute(1, 2, 0).reshape(-1, IN_CHANNELS["image"])
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


class Flux3Unit_ImageIDs(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("image_ids",),
        )

    @staticmethod
    def latent_ids(height, width, device):
        rows, cols = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        zeros = torch.zeros_like(rows)
        ids = torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(-1, 4).float()
        ids[:, 1:3] *= 0.5
        return ids

    def process(self, pipe, height, width):
        return {"image_ids": self.latent_ids(height // 16, width // 16, pipe.device)}


class Flux3Unit_ReferenceEmbedder(PipelineUnit):
    _DOWNSAMPLING_RATIO = 16
    _MATCH_MAX_PIXELS = 4 * 1024 * 1024
    _MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024
    _MIN_SIDE = 64
    _MAX_SIDE = 4096
    _MAX_REFERENCES = 10

    def __init__(self):
        super().__init__(
            input_params=("edit_image",),
            output_params=("reference_tokens", "reference_ids"),
            onload_model_names=("vae",),
        )

    @classmethod
    def reference_canvas(cls, width, height):
        scale = min(1.0, math.sqrt(cls._MATCH_MAX_PIXELS / (width * height)), cls._MAX_SIDE / max(width, height))
        while True:
            fit = (max(1, round(width * scale)), max(1, round(height * scale)))
            canvas = tuple(
                min(cls._MAX_SIDE, max(cls._MIN_SIDE, -(-value // cls._DOWNSAMPLING_RATIO) * cls._DOWNSAMPLING_RATIO))
                for value in fit
            )
            if canvas[0] * canvas[1] <= cls._MATCH_MAX_PIXELS:
                return fit, canvas
            scale *= 0.995

    @classmethod
    def prepare_reference(cls, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        fit, (width, height) = cls.reference_canvas(image.width, image.height)
        if fit != image.size:
            image = image.resize(fit, Image.Resampling.LANCZOS)
        if (width, height) == image.size:
            return image
        left, top = (width - image.width) // 2, (height - image.height) // 2
        padding = ((top, height - image.height - top), (left, width - image.width - left), (0, 0))
        mode = "reflect" if min(image.size) > cls._DOWNSAMPLING_RATIO else "symmetric"
        return Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))

    @classmethod
    def prepare_multi_reference(cls, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if width * height > cls._MULTI_REFERENCE_MAX_PIXELS:
            scale = math.sqrt(cls._MULTI_REFERENCE_MAX_PIXELS / (width * height))
            width, height = int(width * scale), int(height * scale)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        target_width = width // cls._DOWNSAMPLING_RATIO * cls._DOWNSAMPLING_RATIO
        target_height = height // cls._DOWNSAMPLING_RATIO * cls._DOWNSAMPLING_RATIO
        left, top = (width - target_width) // 2, (height - target_height) // 2
        return image.crop((left, top, left + target_width, top + target_height))

    def process(self, pipe, edit_image):
        if edit_image is None or edit_image == []:
            return {"reference_tokens": None, "reference_ids": None}
        references = [edit_image] if isinstance(edit_image, Image.Image) else list(edit_image)
        if len(references) > self._MAX_REFERENCES:
            raise ValueError(f"at most {self._MAX_REFERENCES} references")
        if len(references) > 1:
            prepared = [self.prepare_multi_reference(image) for image in references]
        else:
            prepared = [self.prepare_reference(references[0])]
        pipe.load_models_to_device(self.onload_model_names)
        tokens, ids = [], []
        for index, image in enumerate(prepared):
            tensor = pipe.preprocess_image(image)
            latent = pipe.vae.encode(tensor)
            height, width = latent.shape[-2:]
            token = latent[0].permute(1, 2, 0).reshape(-1, IN_CHANNELS["image_cond"])
            token_ids = Flux3Unit_ImageIDs.latent_ids(height, width, pipe.device)
            token_ids[:, 3] = -(index + 1)
            tokens.append(token.to(pipe.torch_dtype))
            ids.append(token_ids)
        return {"reference_tokens": torch.cat(tokens), "reference_ids": torch.cat(ids)}


class Flux3Unit_PackedBuilder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt_tokens": "prompt_tokens", "prompt_ids": "prompt_ids"},
            input_params_nega={"prompt_tokens": "prompt_tokens", "prompt_ids": "prompt_ids"},
            input_params=("reference_tokens", "reference_ids", "latents", "image_ids"),
            output_params=("packed",),
        )

    def process(self, pipe, prompt_tokens, prompt_ids, reference_tokens, reference_ids, latents, image_ids):
        samples = [[
            (prompt_tokens, prompt_ids),
            None if reference_tokens is None else (reference_tokens, reference_ids),
            (latents, image_ids),
        ]]
        data = {mode: [] for mode in MODES}
        ids = {mode: [] for mode in MODES}
        mode_indices = {mode: [] for mode in MODES}
        counts = dict.fromkeys(MODES, 0)
        for text, condition, image in samples:
            for mode, item in (("text", text), ("image_cond", condition), ("image", image)):
                start = counts[mode]
                if item is not None:
                    data[mode].append(item[0])
                    ids[mode].append(item[1])
                    counts[mode] += item[0].shape[0]
                mode_indices[mode].append((start, counts[mode]))
        device = prompt_tokens.device
        packed_data = {
            mode: torch.cat(data[mode])
            if data[mode]
            else torch.zeros((0, IN_CHANNELS[mode]), device=device, dtype=pipe.torch_dtype)
            for mode in MODES
        }
        packed_ids = {
            mode: torch.cat(ids[mode])
            if ids[mode]
            else torch.zeros((0, 4), device=device, dtype=torch.float32)
            for mode in MODES
        }
        packed = Packed(
            data=packed_data,
            ids=packed_ids,
            timesteps={mode: torch.zeros(packed_data[mode].shape[0], device=device, dtype=pipe.torch_dtype) for mode in MODES},
            mode_indices=mode_indices,
        )
        return {"packed": packed}


def model_fn_flux3(
    dit: Flux3DiT,
    latents,
    timestep,
    packed,
    timesteps_ref=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    packed.data["image"] = latents
    packed.timesteps["image"].fill_((timestep / 1000).item())
    if timesteps_ref is None and dit.flow_map:
        timesteps_ref = timestep / 1000
    model_output = dit(
        packed,
        timesteps_ref=timesteps_ref,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return model_output
