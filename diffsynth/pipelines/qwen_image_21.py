import math
from typing import Union

import torch
from PIL import Image
from tqdm import tqdm

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.qwen_image_21_dit import QwenImage21DiT
from ..models.qwen_image_21_text_encoder import QwenImage21TextEncoder
from ..models.qwen_image_21_vae import QwenImage21VAE


class QwenImage21Pipeline(BasePipeline):
    vae_scale_factor = 16

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=32, width_division_factor=32)
        self.scheduler = FlowMatchScheduler("Qwen-Image")
        self.text_encoder: QwenImage21TextEncoder = None
        self.dit: QwenImage21DiT = None
        self.vae: QwenImage21VAE = None
        self.processor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            QwenImage21Unit_ShapeChecker(),
            QwenImage21Unit_EditImageEmbedder(),
            QwenImage21Unit_PromptEmbedder(),
            QwenImage21Unit_NoiseInitializer(),
            QwenImage21Unit_InputImageEmbedder(),
            QwenImage21Unit_KVCacheInitializer(),
        ]
        self.model_fn = model_fn_qwen_image_21
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = None,
        processor_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = QwenImage21Pipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs or [], vram_limit)
        pipe.text_encoder = model_pool.fetch_model("qwen_image_21_text_encoder")
        pipe.dit = model_pool.fetch_model("qwen_image_21_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_21_vae")
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import AutoProcessor
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str = " ",
        negative_prompt: str = " ",
        cfg_scale: float = 1.0,
        # Editing
        edit_image: Union[Image.Image, list[Image.Image]] = None,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 40,
        # KV cache
        use_kv_cache: bool = True,
        # VAE tiling
        tiled: bool = False,
        tile_size: int = 256,
        tile_stride: int = 192,
        # Progress bar
        progress_bar_cmd=tqdm,
    ):
        # Parameters
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale, "edit_image": edit_image,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "use_kv_cache": use_kv_cache,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, dynamic_shift_len=inputs_shared["latents"].shape[2] * inputs_shared["latents"].shape[3])

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.reshape(1).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id,
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

        # Decode
        self.load_models_to_device(["vae"])
        image = self.vae.decode(inputs_shared["latents"], tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])
        return image


class QwenImage21Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class QwenImage21Unit_PromptEmbedder(PipelineUnit):
    sys_prompt = "Comprehend and analyze the provided prompt."
    prompt_template_t2i = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    prompt_template_ti2i = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image",),
            output_params=("prompt_embeds", "prompt_embeds_mask", "edit_image_pad_mask"),
            onload_model_names=("text_encoder",),
        )
        self._drop_idx = None
        self._img_token_id = None

    @staticmethod
    def _extract_masked_hidden(hidden_states, mask):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    @staticmethod
    def composite_over_white(image):
        if image.mode != "RGBA":
            return image
        # The vision encoder saw the alpha composited over white during training; the VAE still reads all four channels.
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(image, mask=image.getchannel("A"))
        return canvas

    def process(self, pipe, prompt, edit_image):
        if pipe.text_encoder is None or pipe.processor is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if self._drop_idx is None:
            sys_message = [{"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]}]
            sys_tokens = pipe.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
            self._drop_idx = len(sys_tokens) if not sys_tokens or isinstance(sys_tokens[0], int) else len(sys_tokens[0])
            self._img_token_id = pipe.processor.tokenizer.encode("<|image_pad|>")[0]
        # Qwen has no bos token, so an empty string leaves the encoder with nothing to read.
        prompt = [" " if not prompt else prompt]
        if edit_image is None:
            prompts = [self.prompt_template_t2i.format(text) for text in prompt]
        else:
            replacement = "<image1><|vision_start|><|image_pad|><|vision_end|>"
            for index in range(2, len(edit_image) + 1):
                replacement += f" <image{index}><|vision_start|><|image_pad|><|vision_end|>"
            template = self.prompt_template_ti2i.replace("<image1><|vision_start|><|image_pad|><|vision_end|>", replacement)
            prompts = [template.format(text) for text in prompt]

        processor_kwargs = {"text": prompts, "padding": True, "return_tensors": "pt"}
        if edit_image is not None:
            processor_kwargs["images"] = [self.composite_over_white(image) for image in edit_image]
        model_inputs = pipe.processor(**processor_kwargs).to(pipe.device)
        forward_kwargs = {"input_ids": model_inputs.input_ids, "attention_mask": model_inputs.attention_mask}
        if edit_image is not None:
            forward_kwargs.update(pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw)
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids
        hidden_states = pipe.text_encoder(**forward_kwargs)
        split_hidden_states = list(self._extract_masked_hidden(hidden_states, model_inputs.attention_mask))
        split_hidden_states = [hidden_state[self._drop_idx :] for hidden_state in split_hidden_states]
        image_pad_mask = [
            sample_ids[sample_mask.bool()].eq(self._img_token_id)
            for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask)
        ]
        image_pad_mask = [mask[self._drop_idx :] for mask in image_pad_mask]

        attention_masks = [torch.ones(hidden_state.size(0), dtype=torch.long, device=hidden_state.device) for hidden_state in split_hidden_states]
        max_seq_len = max(hidden_state.size(0) for hidden_state in split_hidden_states)
        prompt_embeds = torch.stack([
            torch.cat([hidden_state, hidden_state.new_zeros(max_seq_len - hidden_state.size(0), hidden_state.size(1))])
            for hidden_state in split_hidden_states
        ]).to(dtype=pipe.torch_dtype, device=pipe.device)
        prompt_embeds_mask = torch.stack([torch.cat([mask, mask.new_zeros(max_seq_len - mask.size(0))]) for mask in attention_masks])

        if prompt_embeds_mask.all():
            prompt_embeds_mask = None
        image_pad_mask = torch.stack([torch.cat([mask, mask.new_zeros(max_seq_len - mask.size(0))]) for mask in image_pad_mask])
        return {"prompt_embeds": prompt_embeds, "prompt_embeds_mask": prompt_embeds_mask, "edit_image_pad_mask": image_pad_mask}


class QwenImage21Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe, height, width, seed, rand_device):
        latent_height = 2 * (height // (pipe.vae_scale_factor * 2))
        latent_width = 2 * (width // (pipe.vae_scale_factor * 2))
        noise = pipe.generate_noise((1, 64, latent_height, latent_width), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}


class QwenImage21Unit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("noise", "input_image"),
            output_params=("latents", "input_latents"),
        )

    def process(self, pipe, noise, input_image):
        if getattr(pipe.scheduler, "training", False) and input_image is not None:
            pipe.load_models_to_device(["vae"])
            input_latents = pipe.vae.encode(pipe.preprocess_image(input_image.convert("RGBA")))
            return {"latents": noise, "input_latents": input_latents}
        return {"latents": noise, "input_latents": None}


class QwenImage21Unit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("edit_image", "edit_latents"),
            onload_model_names=("vae",),
        )

    @staticmethod
    def calculate_dimensions(target_area, ratio, min_pixels=None):
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        if min_pixels is not None and width * height < min_pixels:
            beta = math.sqrt(min_pixels / (width * height))
            width = math.ceil(width * beta / 32) * 32
            height = math.ceil(height * beta / 32) * 32
        return width, height

    @staticmethod
    def get_processor_min_pixels(pipe):
        size = getattr(getattr(pipe.processor, "image_processor", None), "size", None)
        return size.get("shortest_edge") if isinstance(size, dict) else getattr(size, "shortest_edge", None)

    def resize_edit_image(self, pipe, edit_image, target_area):
        min_pixels = self.get_processor_min_pixels(pipe)
        return [image.resize(self.calculate_dimensions(target_area, image.size[0] / image.size[1], min_pixels), resample=Image.Resampling.LANCZOS) for image in edit_image]

    def process(self, pipe, edit_image, height, width, tiled, tile_size, tile_stride):
        edit_image = [] if edit_image is None else (edit_image if isinstance(edit_image, list) else [edit_image])
        if len(edit_image) == 0:
            return {"edit_image": None}
        edit_image = [image.convert("RGBA") for image in edit_image]
        edit_image = self.resize_edit_image(pipe, edit_image, height * width)
        pipe.load_models_to_device(self.onload_model_names)
        edit_latents = [pipe.vae.encode(pipe.preprocess_image(image), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride) for image in edit_image]
        return {"edit_image": edit_image, "edit_latents": edit_latents}


class QwenImage21Unit_KVCacheInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("use_kv_cache", "cfg_scale"),
            output_params=("kv_cache",),
        )

    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        if getattr(pipe.scheduler, "training", False) or not inputs_shared["use_kv_cache"]:
            inputs_posi["kv_cache"] = None
            inputs_nega["kv_cache"] = None
            return inputs_shared, inputs_posi, inputs_nega
        inputs_posi["kv_cache"] = [{} for _ in pipe.dit.transformer_blocks]
        if inputs_shared["cfg_scale"] != 1.0:
            inputs_nega["kv_cache"] = [{} for _ in pipe.dit.transformer_blocks]
        return inputs_shared, inputs_posi, inputs_nega


def model_fn_qwen_image_21(
    dit: QwenImage21DiT,
    latents,
    timestep,
    prompt_embeds,
    prompt_embeds_mask,
    edit_image_pad_mask,
    edit_latents=None,
    kv_cache=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    latent_height, latent_width = latents.shape[2], latents.shape[3]
    img_shapes = [[*[(1, edit_latent.shape[2], edit_latent.shape[3]) for edit_latent in edit_latents or []], (1, latent_height, latent_width)]]
    latents = patchify(latents)
    target_seq_len = latents.shape[1]
    target_image_pad_mask = torch.ones((1, target_seq_len // 4), dtype=torch.bool, device=latents.device)
    image_pad_mask = torch.cat([edit_image_pad_mask, target_image_pad_mask], dim=1)
    if edit_latents is not None:
        edit_latents = torch.cat([patchify(edit_latent) for edit_latent in edit_latents], dim=1)
    hidden_states = latents if edit_latents is None else torch.cat([edit_latents, latents], dim=1)
    model_output = dit(
        hidden_states=hidden_states,
        timestep=timestep / 1000,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        img_shapes=img_shapes,
        img_mask=image_pad_mask,
        kv_cache=kv_cache,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return unpatchify(model_output[:, -target_seq_len:], latent_height, latent_width)


def patchify(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    return latents.view(batch_size, num_channels_latents, height * width).transpose(1, 2)


def unpatchify(latents, height, width):
    batch_size, _, channels = latents.shape
    return latents.transpose(1, 2).reshape(batch_size, channels, height, width)
