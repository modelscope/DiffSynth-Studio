import functools
import glob
import json
import math
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from einops import rearrange

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.fibo_dit import FiboDiT
from ..models.flux2_vae import Flux2VAE


DEFAULT_NEGATIVE_PROMPT = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)
VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
TEXT_ENCODER_OUT_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
TEXT_ENCODER_IMAGE_SIZE = 768
DEFAULT_IMAGE_AREA = 1024 * 1024


class FiboPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=16, width_division_factor=16)
        self.scheduler = FlowMatchScheduler("Fibo")
        self.text_encoder = None
        self.dit: FiboDiT = None
        self.vae: Flux2VAE = None
        self.processor = None
        self.default_fonts = []
        self.in_iteration_models = ("dit",)
        self.units = [
            FiboUnit_ShapeChecker(),
            FiboUnit_EditImagePreprocess(),
            FiboUnit_PromptEmbedder(),
            FiboUnit_NoiseInitializer(),
            FiboUnit_InputImageEmbedder(),
            FiboUnit_EditImageEmbedder(),
        ]
        self.model_fn = model_fn_fibo
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="processor/"),
        font_config: ModelConfig = ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="fonts/"),
        vram_limit: float = None,
    ):
        pipe = FiboPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        # Fibo and Krea-2 share the same Qwen3-VL-4B text encoder structure.
        pipe.text_encoder = model_pool.fetch_model("krea2_text_encoder")
        pipe.dit = model_pool.fetch_model("fibo_dit")
        pipe.vae = model_pool.fetch_model("flux2_vae")
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen3VLProcessor
            pipe.processor = Qwen3VLProcessor.from_pretrained(processor_config.path)
        if font_config is not None:
            font_config.download_if_necessary()
            if isinstance(font_config.path, list):
                pipe.default_fonts = sorted(
                    path for path in font_config.path if path.lower().endswith((".otf", ".ttf"))
                )
            elif os.path.isdir(font_config.path):
                pipe.default_fonts = sorted(
                    path for path in glob.glob(os.path.join(font_config.path, "*"))
                    if path.lower().endswith((".otf", ".ttf"))
                )
            else:
                pipe.default_fonts = [font_config.path]
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    def decode_latents(self, latents):
        latents = rearrange(latents, "B C (H P) (W Q) -> B (C P Q) H W", P=2, Q=2)
        return self.vae.decode(latents.to(device=self.device, dtype=self.torch_dtype))

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = None,
        edit_image: Image.Image = None,
        mask: Image.Image = None,
        text_images: list = None,
        fonts: list = None,
        render_text_images: bool = True,
        height: int = None,
        width: int = None,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 30,
        denoising_strength: float = 1.0,
        seed: int = None,
        rand_device: str = "cpu",
        progress_bar_cmd=tqdm,
    ):
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale, "edit_image": edit_image, "mask": mask,
            "text_images": text_images, "fonts": fonts, "render_text_images": render_text_images,
            "height": height, "width": width, "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps, "denoising_strength": denoising_strength,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=(inputs_shared["height"] // 16) * (inputs_shared["width"] // 16))

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id,
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        self.load_models_to_device(["vae"])
        image = rearrange(inputs_shared["latents"], "B C (H P) (W Q) -> B (C P Q) H W", P=2, Q=2)
        image = self.vae.decode(image.to(device=self.device, dtype=self.torch_dtype))
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])
        return image


class FiboUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "edit_image"), output_params=("height", "width"))

    @staticmethod
    def get_image_size(aspect_ratio, area=DEFAULT_IMAGE_AREA, multiple=16):
        min_side = 4 * multiple
        sizes = [
            (width, height)
            for width in range(min_side, 2 * area // min_side, multiple)
            for height in range(min_side, 2 * area // width, multiple)
            if 0.9 * area <= width * height <= 1.1 * area
        ]
        def crop(size):
            ratio = size[0] / size[1]
            return 1 - min(ratio / aspect_ratio, aspect_ratio / ratio)
        return min(sizes, key=lambda size: (round(crop(size), 9), abs(size[0] * size[1] - area), -size[0]))

    def process(self, pipe, height, width, edit_image):
        if edit_image is not None:
            if not isinstance(edit_image, (Image.Image, list)):
                raise ValueError("`edit_image` must be a PIL image or a list of PIL images.")
            if isinstance(edit_image, list) and not 1 <= len(edit_image) <= 5:
                raise ValueError(f"`edit_image` must hold 1 to 5 images but holds {len(edit_image)}.")
            if height is None and width is None:
                first = edit_image[0] if isinstance(edit_image, list) else edit_image
                width, height = self.get_image_size(first.width / first.height)
        height, width = pipe.check_resize_height_width(height or 1024, width or 1024)
        return {"height": height, "width": width}


class FiboUnit_EditImagePreprocess(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "mask", "height", "width"),
            output_params=("vae_images", "text_encoder_images"),
        )

    @staticmethod
    def resize_and_crop(image, width, height):
        image = image.convert("RGB")
        scale = max(height / image.height, width / image.width)
        resized_width, resized_height = math.ceil(image.width * scale), math.ceil(image.height * scale)
        image = image.resize((resized_width, resized_height), resample=Image.LANCZOS)
        left, top = (resized_width - width) // 2, (resized_height - height) // 2
        return image.crop((left, top, left + width, top + height))

    @staticmethod
    def grey_out(image, mask):
        grey = Image.new("RGB", image.size, (128, 128, 128))
        binary_mask = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
        return Image.composite(grey, image.convert("RGB"), binary_mask)

    def process(self, pipe, edit_image, mask, height, width):
        if edit_image is None:
            return {}
        images = edit_image if isinstance(edit_image, list) else [edit_image]
        if mask is not None:
            if len(images) != 1:
                raise ValueError("`mask` needs exactly one image in `edit_image`.")
            if mask.size != images[0].size:
                raise ValueError(f"`mask` has size {mask.size} but `edit_image` has size {images[0].size}.")
            images = [self.grey_out(images[0], mask)]
        if len(images) == 1:
            width, height = self.get_size(images[0], height, width)
            sizes = [(width, height)]
        else:
            sizes = [self.get_size(image) for image in images]
        crops = [self.resize_and_crop(image, size[0], size[1]) for image, size in zip(images, sizes)]
        text_images = []
        for crop in crops:
            text_image = crop.copy()
            text_image.thumbnail((TEXT_ENCODER_IMAGE_SIZE, TEXT_ENCODER_IMAGE_SIZE))
            text_images.append(text_image)
        return {"vae_images": crops, "text_encoder_images": text_images}

    @staticmethod
    def get_size(image, height=None, width=None):
        if height is not None or width is not None:
            return width, height
        return FiboUnit_ShapeChecker.get_image_size(image.width / image.height)


class FiboUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True, onload_model_names=("text_encoder",))

    @staticmethod
    def parse_prompt(prompt):
        try:
            caption = json.loads(prompt)
        except (json.JSONDecodeError, TypeError):
            return prompt, None
        if not isinstance(caption, dict):
            return prompt, None
        instruction = caption.pop("edit_instruction", None)
        if "structured_caption" in caption:
            caption = caption.pop("structured_caption")
            if not isinstance(caption, dict):
                return caption or None, instruction

        def empty(value):
            if isinstance(value, float):
                return math.isnan(value)
            return value is None or (isinstance(value, (str, dict, list)) and not value)

        def clean(value):
            if isinstance(value, dict):
                cleaned = {}
                for key, item in value.items():
                    item = clean(item)
                    if not empty(item):
                        cleaned[key] = item
                return cleaned
            if isinstance(value, list):
                cleaned = []
                for item in value:
                    item = clean(item)
                    if not empty(item):
                        cleaned.append(item)
                return cleaned
            return value

        keys = set(caption)
        scores = ("aesthetic_score", "preference_score")
        caption = {k: v for k, v in clean(caption).items() if k not in scores}
        if isinstance(caption, dict):
            caption = {k: {i: j for i, j in v.items() if i not in scores} if isinstance(v, dict) else v for k, v in caption.items()}
            caption = {k: v for k, v in caption.items() if v != {}}
        if keys == {"short_description", "text_render"}:
            caption = {"short_description": caption.get("short_description") or "", "text_render": caption.get("text_render") or []}
        return caption or None, instruction

    @staticmethod
    def mark_text_images(caption, marker=VISION_MARKER):
        for entry in caption.get("text_render") or []:
            if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]:
                entry["image_tokens"] = marker
        return caption

    @classmethod
    def format_generation_prompt(cls, prompt):
        caption, _ = cls.parse_prompt(prompt)
        return caption if isinstance(caption, str) else json.dumps(caption or {}, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def format_edit_prompt(cls, prompt, num_images, use_text_images=False):
        caption, instruction = cls.parse_prompt(prompt)
        if instruction is None:
            caption, instruction = None, prompt
        if use_text_images and isinstance(caption, dict):
            cls.mark_text_images(caption)
        edit = {"image": VISION_MARKER} if num_images == 1 else {f"image_{k}": VISION_MARKER for k in range(1, num_images + 1)}
        if caption:
            edit["structured_caption"] = caption
        edit["edit_instruction"] = instruction
        return json.dumps(edit, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def load_font(path):
        return ImageFont.truetype(path, 20, layout_engine=ImageFont.Layout.BASIC)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def default_font():
        font = ImageFont.load_default(20)
        if not isinstance(font, ImageFont.FreeTypeFont):
            raise ImportError("Drawing text images needs Pillow 10.1 or later, built with FreeType.")
        return font

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def has_glyph(font, character):
        try:
            notdef = font.getmask("\ufffe")
            mask = font.getmask(character)
        except Exception:
            return True
        return not (mask.size == notdef.size and bytes(mask) == bytes(notdef))

    @classmethod
    def pick_font(cls, text, fonts):
        candidates = [cls.default_font(), *fonts]
        for font in candidates:
            if all(ord(character) < 128 or cls.has_glyph(font, character) for character in text):
                return font
        return candidates[0]

    @classmethod
    def draw_text_image(cls, text, fonts):
        font = cls.pick_font(text, fonts)
        line = " ".join(text.split())
        lines = [line[i:i + 256] for i in range(0, len(line), 256)] or [""]
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
        boxes = [draw.textbbox((0, 0), line or " ", font=font) for line in lines]
        widths = [box[2] - box[0] for box in boxes]
        heights = [max(1, box[3] - box[1]) for box in boxes]
        image = Image.new("RGB", (max(widths) + 4, sum(heights) + 4 * len(lines)), "white")
        draw = ImageDraw.Draw(image)
        y = 2
        for line, box, height in zip(lines, boxes, heights):
            draw.text((2 - box[0], y - box[1]), line, font=font, fill=(0, 0, 0))
            y += height + 4
        return image

    @classmethod
    def text_images_from_prompt(cls, prompt, fonts):
        caption, _ = cls.parse_prompt(prompt)
        if not isinstance(caption, dict):
            return []
        return [
            cls.draw_text_image(entry["text"], fonts)
            for entry in caption.get("text_render") or []
            if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]
        ]

    @classmethod
    def encode_prompt(cls, text_encoder, processor, prompt, image=None, text_images=None, device=None):
        text_images = text_images or []
        if image is not None:
            prompt = cls.format_edit_prompt(prompt, len(image), use_text_images=bool(text_images))
            images = list(image)
        elif text_images:
            try:
                caption = json.loads(prompt)
            except json.JSONDecodeError as error:
                raise ValueError("`text_images` requires a JSON caption prompt.") from error
            prompt = json.dumps(cls.mark_text_images(caption), separators=(",", ":"), ensure_ascii=False)
            images = []
        else:
            images = []
        images += text_images

        text_inputs = processor(
            text=[PROMPT_TEMPLATE.format(prompt)], images=images or None,
            padding="longest", max_length=4096, truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden_states = text_encoder(**text_inputs)
        dtype = next(iter(hidden_states)).dtype
        normalized = {}
        for layer in sorted({layer for layers in TEXT_ENCODER_OUT_LAYERS for layer in layers}):
            states = hidden_states[layer].float()
            normalized[layer] = (states * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(dtype)
        bundles = [torch.cat([normalized[layer] for layer in layers], dim=-1) for layers in TEXT_ENCODER_OUT_LAYERS]
        mask = text_inputs.attention_mask.bool()
        return torch.stack(bundles, dim=1), (None if mask.all() else mask)

    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        text_images = inputs_shared.get("text_images")
        fonts = inputs_shared.get("fonts") or pipe.default_fonts
        fallback_fonts = [self.load_font(path) for path in fonts]
        if text_images is None and inputs_shared.get("render_text_images", False):
            text_images = self.text_images_from_prompt(inputs_posi["prompt"], fallback_fonts)
        pipe.load_models_to_device(self.onload_model_names)

        prompt = inputs_posi["prompt"]
        prompt = self.format_generation_prompt(prompt) if inputs_shared.get("edit_image") is None else prompt
        inputs_posi["prompt_embeds"], inputs_posi["prompt_embeds_mask"] = self.encode_prompt(
            pipe.text_encoder, pipe.processor, prompt,
            image=inputs_shared.get("text_encoder_images"), text_images=text_images, device=pipe.device,
        )
        if inputs_shared["cfg_scale"] != 1.0:
            negative_prompt = inputs_nega.get("prompt") or DEFAULT_NEGATIVE_PROMPT
            inputs_nega["prompt_embeds"], inputs_nega["prompt_embeds_mask"] = self.encode_prompt(
                pipe.text_encoder, pipe.processor, negative_prompt, device=pipe.device
            )
        return inputs_shared, inputs_posi, inputs_nega


class FiboUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "seed", "rand_device"), output_params=("noise",))

    def process(self, pipe, height, width, seed, rand_device):
        return {"noise": pipe.generate_noise(
            (1, 32, height // 8, width // 8), seed=seed, rand_device=rand_device, rand_torch_dtype=torch.float32
        )}


class FiboUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "height", "width"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, input_image, noise, height, width):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(FiboUnit_EditImagePreprocess.resize_and_crop(input_image, width, height))
        return {"latents": noise, "input_latents": FiboUnit_EditImageEmbedder.encode_image(pipe, image)}


class FiboUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vae_images",),
            output_params=("context_latents",),
            onload_model_names=("vae",),
        )

    @staticmethod
    def encode_image(pipe, image):
        if isinstance(image, Image.Image):
            image = pipe.preprocess_image(image)
        image = image.to(device=pipe.device, dtype=pipe.torch_dtype).contiguous()
        return rearrange(pipe.vae.encode(image), "B (C P Q) H W -> B C (H P) (W Q)", P=2, Q=2)

    def process(self, pipe, vae_images):
        if vae_images is None:
            return {"context_latents": None}
        pipe.load_models_to_device(self.onload_model_names)
        return {"context_latents": [
            self.encode_image(pipe, image).to(pipe.torch_dtype) for image in vae_images
        ]}


def model_fn_fibo(
    dit: FiboDiT,
    latents=None,
    timestep=None,
    prompt_embeds=None,
    prompt_embeds_mask=None,
    context_latents=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    return dit(
        latents.to(prompt_embeds.dtype),
        timestep=timestep / 1000,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_embeds_mask,
        context_latents=context_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    ).float()
