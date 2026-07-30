import json
import math

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, Union

from ..core.device.npu_compatible_device import get_device_type
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..diffusion.flow_match import FlowMatchScheduler

from ..models.lingbot_video_dit import LingBotVideoDiT
from ..models.krea2_text_encoder import Krea2TextEncoder
from ..models.qwen_image_vae import QwenImageVAE


class LingBotVideoPipeline(BasePipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
            time_division_factor=4, time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler(template="Wan")
        self.text_encoder: Krea2TextEncoder = None
        self.dit: LingBotVideoDiT = None
        self.vae: QwenImageVAE = None
        self.processor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            LingBotVideoUnit_ShapeChecker(),
            LingBotVideoUnit_NoiseInitializer(),
            LingBotVideoUnit_InputVideoEmbedder(),
            LingBotVideoUnit_ImageEmbedder(),
            LingBotVideoUnit_PromptEmbedder(),
        ]
        self.model_fn = model_fn_lingbot_video
        self.compilable_models = ["dit"]
        self.default_negative_prompt = (
            '{"universal_negative": {'
            '"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], '
            '"artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], '
            '"composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], '
            '"temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], '
            '"material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'
        )
        self.default_negative_prompt_image = (
            '{"universal_negative": {'
            '"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], '
            '"artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], '
            '"composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], '
            '"material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'
        )

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = LingBotVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("krea2_text_encoder")
        pipe.dit = model_pool.fetch_model("lingbot_video_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_vae")

        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import AutoProcessor
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Structured caption (dict) or a plain string.
        prompt: Union[str, dict] = "",
        negative_prompt: Union[str, dict] = "",
        # Image-to-video (TI2V)
        input_image: Image.Image = None,
        # Video-to-video
        input_video: list[Image.Image] = None,
        denoising_strength: float = 1.0,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Shape
        height: int = 480,
        width: int = 480,
        num_frames: int = 81,
        # Classifier-free guidance
        cfg_scale: float = 3.0,
        # Scheduler
        num_inference_steps: int = 40,
        sigma_shift: float = 3.0,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Inputs
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "input_image": input_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"].to(dtype=self.torch_dtype, device=self.device)
        video = self.vae.decode_video(latents)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


class LingBotVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: LingBotVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class LingBotVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: LingBotVideoPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.dit.in_channels, length, height // 8, width // 8)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class LingBotVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: LingBotVideoPipeline, input_video, noise):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode_video(video).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


class LingBotVideoUnit_ImageEmbedder(PipelineUnit):
    # Qwen3-VL vision token-budget bounds used by smart_resize (official defaults).
    IMAGE_MIN_TOKEN_NUM = 4
    IMAGE_MAX_TOKEN_NUM = 16384
    MAX_RATIO = 200
    SPATIAL_MERGE_SIZE = 2

    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width"),
            output_params=("latents", "first_frame_latents", "vlm_image"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: LingBotVideoPipeline, input_image, latents, height, width):
        if input_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        # (1, C, 1, H, W) in [0, 1] -> [-1, 1] to match the VAE input range; encode_video applies
        # the latent normalisation, giving the clean cond latent the DiT was trained to inpaint on.
        pixel = self.preprocess_cond_image(input_image, height, width)
        pixel = pixel.to(dtype=pipe.torch_dtype, device=pipe.device)
        first_frame_latents = pipe.vae.encode_video(pixel * 2.0 - 1.0).to(dtype=pipe.torch_dtype, device=pipe.device)
        vlm_image = self.vlm_image(pipe, pixel)
        # Pin the clean condition latent into the first temporal slot before sampling.
        cond_t = first_frame_latents.shape[2]
        latents[:, :, :cond_t] = first_frame_latents
        return {"latents": latents, "first_frame_latents": first_frame_latents, "vlm_image": vlm_image}

    @staticmethod
    def preprocess_cond_image(image: Image.Image, height, width) -> torch.Tensor:
        # TI2V condition frame -> (1, C, 1, H, W) pixel tensor in [0, 1], aspect-ratio
        # preserving cover-resize + center-crop to (height, width).
        raw = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).contiguous()
        old_h, old_w = raw.shape[-2:]
        scale = max(height / old_h, width / old_w)
        new_h = max(math.ceil(old_h * scale), height)
        new_w = max(math.ceil(old_w * scale), width)
        resized = F.interpolate(raw.float(), size=(new_h, new_w), mode="bilinear", align_corners=False)
        top = int(round((new_h - height) / 2.0))
        left = int(round((new_w - width) / 2.0))
        cropped = resized[:, :, top: top + height, left: left + width] / 255.0
        return cropped.unsqueeze(2)

    def vlm_image(self, pipe: LingBotVideoPipeline, pixel: torch.Tensor) -> Image.Image:
        # Build the PIL image handed to the text encoder from the condition pixel tensor,
        # smart-resized to the Qwen3-VL patch grid.
        image = self._pixel_tensor_to_pil(pixel)
        patch_factor = self._vision_patch_size(pipe) * self.SPATIAL_MERGE_SIZE
        w, h = image.size
        resized_height, resized_width = self.smart_resize(h, w, factor=patch_factor)
        return image.resize((resized_width, resized_height))

    @staticmethod
    def _vision_patch_size(pipe: LingBotVideoPipeline) -> int:
        # Resolve the Qwen3-VL vision patch size, used with SPATIAL_MERGE_SIZE to pick the
        # smart-resize grid factor.
        for obj in (
            getattr(getattr(pipe.text_encoder, "config", None), "vision_config", None),
            getattr(getattr(pipe.processor, "image_processor", None), "config", None),
            getattr(pipe.processor, "image_processor", None),
        ):
            patch = getattr(obj, "patch_size", None)
            if patch is not None:
                return int(patch)
        return 16

    @staticmethod
    def _pixel_tensor_to_pil(pixel: torch.Tensor) -> Image.Image:
        # Match torchvision.transforms.ToPILImage for a float CHW image in [0, 1].
        # pixel: (B, C, T, H, W); take batch 0, temporal slot 0.
        frame = pixel[0, :, 0].detach().cpu().clamp(0, 1)
        array = frame.permute(1, 2, 0).mul(255).byte().numpy()
        return Image.fromarray(array, mode="RGB")

    @classmethod
    def smart_resize(cls, height: int, width: int, factor: int,
                     min_pixels: Optional[int] = None, max_pixels: Optional[int] = None):
        # Resize so both sides are multiples of `factor` and the token count stays in
        # [min_pixels, max_pixels] while preserving aspect ratio. Ported verbatim from the
        # official LingBot-Video i2v pipeline (Qwen3-VL smart-resize).
        max_pixels = max_pixels if max_pixels is not None else cls.IMAGE_MAX_TOKEN_NUM * factor**2
        min_pixels = min_pixels if min_pixels is not None else cls.IMAGE_MIN_TOKEN_NUM * factor**2
        if max_pixels < min_pixels:
            raise ValueError("max_pixels must be greater than or equal to min_pixels.")
        if max(height, width) / min(height, width) > cls.MAX_RATIO:
            raise ValueError(f"absolute aspect ratio must be smaller than {cls.MAX_RATIO}.")
        resized_height = max(factor, cls._round_by_factor(height, factor))
        resized_width = max(factor, cls._round_by_factor(width, factor))
        if resized_height * resized_width > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            resized_height = cls._floor_by_factor(height / beta, factor)
            resized_width = cls._floor_by_factor(width / beta, factor)
        elif resized_height * resized_width < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            resized_height = cls._ceil_by_factor(height * beta, factor)
            resized_width = cls._ceil_by_factor(width * beta, factor)
        return resized_height, resized_width

    @staticmethod
    def _round_by_factor(number: float, factor: int) -> int:
        return round(number / factor) * factor

    @staticmethod
    def _ceil_by_factor(number: float, factor: int) -> int:
        return math.ceil(number / factor) * factor

    @staticmethod
    def _floor_by_factor(number: float, factor: int) -> int:
        return math.floor(number / factor) * factor


class LingBotVideoUnit_PromptEmbedder(PipelineUnit):
    TOKEN_LENGTH = 37698
    HIDDEN_STATE_SKIP_LAYER = 0
    # Token block prepended to the prompt (TI2V) so the encoder attends to the condition frame.
    IMG_PROMPT_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"
    PROMPT_TEMPLATE = (
        "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
        "a text prompt with an image reference, or a text prompt with a video reference "
        "or a video reference alone, generate an \"Enhanced prompt\" that provides detailed "
        "visual descriptions suitable for video generation. Evaluate the level of detail "
        "in the user's input: if it is simple, enrich it by adding specifics about colors, "
        "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
        "progression, and spatial relationships to create vivid, concrete, and temporally "
        "coherent scenes to create vivid and concrete scenes. Please generate only the "
        "enhanced description for the prompt below and avoid including any additional "
        "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    _RUNTIME_KEYS = {"duration", "fps", "height", "width", "num_frames", "resolution", "ratio"}

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("vlm_image",),
            output_params=("context", "encoder_attention_mask"),
            onload_model_names=("text_encoder", ),
        )
        self._crop_start: Optional[int] = None

    @classmethod
    def normalize_caption(cls, prompt):
        if isinstance(prompt, dict):
            if "caption" in prompt:
                caption = prompt["caption"]
            else:
                caption = {k: v for k, v in prompt.items() if k not in cls._RUNTIME_KEYS}
            if isinstance(caption, (dict, list)):
                return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
            return str(caption)
        elif isinstance(prompt, str):
            return prompt
        else:
            raise TypeError(f"prompt must be a str or a dict, not {type(prompt)}")

    def _compute_crop_start(self, pipe: LingBotVideoPipeline) -> int:
        if self._crop_start is None:
            marker = "<|USER_INPUT_MARKER|>"
            marked = self.PROMPT_TEMPLATE.format(marker)
            marker_pos = marked.find(marker)
            if marker_pos < 0:
                self._crop_start = 0
            else:
                prefix = pipe.processor(
                    text=marked[:marker_pos],
                    images=None,
                    videos=None,
                    return_tensors="pt",
                )
                self._crop_start = int(prefix["input_ids"].shape[1])
        return self._crop_start

    def encode_prompt(self, pipe: LingBotVideoPipeline, prompt, vlm_image=None):
        prompt = self.normalize_caption(prompt)
        # TI2V: prepend the image-token block so the encoder attends to the condition frame.
        # The image tokens land after the template prefix, so crop_start is unaffected.
        visual_template = self.IMG_PROMPT_TEMPLATE if vlm_image is not None else ""
        text = self.PROMPT_TEMPLATE.format(visual_template + prompt)
        inputs = pipe.processor(
            text=[text],
            images=[vlm_image] if vlm_image is not None else None,
            videos=None,
            do_resize=False,
            truncation=True,
            max_length=self.TOKEN_LENGTH,
            padding="longest",
            return_tensors="pt",
        )
        inputs = inputs.to(pipe.device)
        hidden_states = pipe.text_encoder(**inputs)
        prompt_embeds = hidden_states[-(self.HIDDEN_STATE_SKIP_LAYER + 1)]
        prompt_mask = inputs["attention_mask"]

        crop_start = self._compute_crop_start(pipe)
        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_mask = prompt_mask[:, crop_start:]

        if prompt_embeds.shape[0] == 1:
            true_len = int(prompt_mask[0].sum().item())
            prompt_embeds = prompt_embeds[:, :true_len]
            prompt_mask = prompt_mask[:, :true_len]

        return prompt_embeds.to(dtype=pipe.torch_dtype), prompt_mask

    def process(self, pipe: LingBotVideoPipeline, prompt, vlm_image=None) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds, prompt_mask = self.encode_prompt(pipe, prompt, vlm_image=vlm_image)
        return {"context": prompt_embeds, "encoder_attention_mask": prompt_mask}


def model_fn_lingbot_video(
    dit: LingBotVideoDiT,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    first_frame_latents: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    if first_frame_latents is not None:
        latents[:, :, :first_frame_latents.shape[2]] = first_frame_latents
    noise_pred = dit(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=context,
        encoder_attention_mask=encoder_attention_mask,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return noise_pred
