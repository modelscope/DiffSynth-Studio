import torch, math
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..diffusion.flow_match import FlowMatchScheduler
from ..models.minimax_h3_dit import patchify_video, unpatchify_video
from ..models.minimax_h3_image_dit import MiniMaxH3ImageDiT, MiniMaxH3ImagePDDHeads
from ..models.minimax_h3_image_vae import MiniMaxH3ImageVAE
from ..models.minimax_h3_text_encoder import MiniMaxH3TextEncoder


# The transformer patchifies the latents with a 2x2 patch, so a row of the packed sequence covers 32x32 pixels.
MINIMAX_H3_IMAGE_PATCH_SIZE = (1, 2, 2)
MINIMAX_H3_IMAGE_LATENT_CHANNELS = 24
MINIMAX_H3_IMAGE_CANVAS_MULTIPLE = 32
MINIMAX_H3_IMAGE_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_IMAGE_PIXEL_STD = (0.229, 0.224, 0.225)
MINIMAX_H3_IMAGE_TEXT_TAG = 1
MINIMAX_H3_IMAGE_IMAGE_TAG = 0
MINIMAX_H3_IMAGE_ROPE_SPATIAL_SCALE = 32
MINIMAX_H3_IMAGE_MAX_SEQUENCE_LENGTH = 4096
MINIMAX_H3_IMAGE_MIN_GENERATION_AREA = 2048 * 2048
MINIMAX_H3_IMAGE_NUM_INFERENCE_STEPS = 6
MINIMAX_H3_IMAGE_REFERENCE_NOISE_AUG = 0.999
MINIMAX_H3_IMAGE_REFERENCE_MIN_SHORT_EDGE = 1024
MINIMAX_H3_IMAGE_REFERENCE_MAX_SHORT_EDGE = 2048
MINIMAX_H3_IMAGE_VISION_MIN_SHORT_EDGE = 1024
MINIMAX_H3_IMAGE_VISION_MAX_SHORT_EDGE = 1024
# The sigma grid the checkpoint's six output heads were distilled on, one head per grid point.
MINIMAX_H3_IMAGE_SAMPLE_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132)


class MiniMaxH3ImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=32, width_division_factor=32)
        self.scheduler = FlowMatchScheduler("MiniMax-H3-Image")
        self.text_encoder: MiniMaxH3TextEncoder = None
        self.dit: MiniMaxH3ImageDiT = None
        self.pdd_heads: MiniMaxH3ImagePDDHeads = None
        self.vae: MiniMaxH3ImageVAE = None
        self.tokenizer = None
        self.processor = None
        self.in_iteration_models = ("dit", "pdd_heads")
        self.units = [
            MiniMaxH3ImageUnit_ShapeChecker(),
            MiniMaxH3ImageUnit_PromptEmbedder(),
            MiniMaxH3ImageUnit_InputImageEmbedder(),
            MiniMaxH3ImageUnit_NoiseInitializer(),
            MiniMaxH3ImageUnit_ReferenceEncoder(),
            MiniMaxH3ImageUnit_PackedSequenceBuilder(),
            MiniMaxH3ImageUnit_TimestepPlanner(),
        ]
        self.model_fn = model_fn_minimax_h3_image
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = MiniMaxH3ImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("minimax_h3_text_encoder")
        pipe.dit = model_pool.fetch_model("minimax_h3_image_dit")
        pipe.pdd_heads = model_pool.fetch_model("minimax_h3_image_pdd_heads")
        pipe.vae = model_pool.fetch_model("minimax_h3_image_vae")
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import AutoProcessor, AutoTokenizer
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
            pipe.tokenizer = AutoTokenizer.from_pretrained(processor_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    def decode_latents(self, latents, tiling=True):
        """Decodes `[1, C, 1, H, W]` latents into an image in the `[0, 1]` range."""
        latents_mean = self.vae.latents_mean.to(device=self.device, dtype=torch.float32).view(1, -1, 1, 1)
        latents_std = self.vae.latents_std.to(device=self.device, dtype=torch.float32).view(1, -1, 1, 1)
        latents = latents[:, :, 0].to(torch.float32) * latents_std + latents_mean
        image = self.vae.decode(latents, tiling=tiling)
        pixel_mean = torch.tensor(MINIMAX_H3_IMAGE_PIXEL_MEAN, device=self.device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_IMAGE_PIXEL_STD, device=self.device).view(1, -1, 1, 1)
        image = (image.float() * pixel_std + pixel_mean).clamp(0, 1)
        # Quantize to 8 bits by truncation, as the released pipeline does, before any resize.
        return self.vae_output_to_image(image[:, :, None], pattern="B C T H W", min_value=0, max_value=1)

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str = "",
        max_sequence_length: int = MINIMAX_H3_IMAGE_MAX_SEQUENCE_LENGTH,
        # Image editing
        images: Image.Image = None,
        # Shape
        height: int = 2048,
        width: int = 2048,
        min_generation_area: int = MINIMAX_H3_IMAGE_MIN_GENERATION_AREA,
        tiling: bool = True,
        # Randomness
        seed: int = 0,
        rand_device: str = "cpu",
        latents: torch.Tensor = None,
        reference_noise: list = None,
        # Steps
        num_inference_steps: int = MINIMAX_H3_IMAGE_NUM_INFERENCE_STEPS,
        distilled: bool = True,
        # Progress bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, distilled=distilled)

        # Parameters
        inputs_posi = {"prompt": prompt}
        inputs_nega = {}
        inputs_shared = {
            "prompt": prompt,
            "images": images,
            "height": height, "width": width, "min_generation_area": min_generation_area,
            "max_sequence_length": max_sequence_length,
            "seed": seed, "rand_device": rand_device,
            "latents": latents, "reference_noise": reference_noise,
            "tiling": tiling,
        }

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            unique_timesteps, timestep_indices = inputs_shared["row_timestep_plan"][progress_id]
            noise_pred = self.model_fn(
                **models, **inputs_shared,
                timestep=timestep,
                unique_timesteps=unique_timesteps,
                timestep_indices=timestep_indices,
                pdd_step_index=progress_id,
            )
            inputs_shared["latents"] = self.step(self.scheduler, inputs_shared["latents"], progress_id, noise_pred=noise_pred)

        # Decode
        self.load_models_to_device(["vae"])
        image = self.decode_latents(inputs_shared["latents"], tiling=tiling)
        if image.size != (inputs_shared["output_width"], inputs_shared["output_height"]):
            image = image.resize((inputs_shared["output_width"], inputs_shared["output_height"]), Image.LANCZOS)
        self.load_models_to_device([])
        return image


class MiniMaxH3ImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "min_generation_area"),
            output_params=("height", "width", "output_height", "output_width", "latent_height", "latent_width"),
        )

    @staticmethod
    def get_canvas_size(height, width, min_generation_area=MINIMAX_H3_IMAGE_MIN_GENERATION_AREA, multiple=MINIMAX_H3_IMAGE_CANVAS_MULTIPLE):
        # An image below `min_generation_area` pixels is generated larger at the same aspect ratio, and both sides of
        # the canvas round to the nearest multiple of 32. The requested size is kept as the output size.
        scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
        return (
            max(multiple, round(height * scale / multiple) * multiple),
            max(multiple, round(width * scale / multiple) * multiple),
        )

    def process(self, pipe: MiniMaxH3ImagePipeline, height, width, min_generation_area):
        height, width = pipe.check_resize_height_width(height, width)
        if min_generation_area is None:
            min_generation_area = MINIMAX_H3_IMAGE_MIN_GENERATION_AREA
        canvas_height, canvas_width = self.get_canvas_size(height, width, min_generation_area)
        ratio = pipe.vae.spatial_compression_ratio
        return {
            "height": canvas_height, "width": canvas_width,
            "output_height": height, "output_width": width,
            "latent_height": canvas_height // ratio, "latent_width": canvas_width // ratio,
        }


class MiniMaxH3ImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("prompt", "images", "max_sequence_length"),
            output_params=("prompt_embeds", "text_token_tags", "reference_images"),
            onload_model_names=("text_encoder",),
        )

    def resize_reference_image(self, image, min_short_edge, max_short_edge):
        # A reference image is not resized when its short edge is within the consumer's bounds; otherwise it is
        # scaled, keeping its ratio, so that the short edge lands on the nearer bound.
        multiple = MINIMAX_H3_IMAGE_CANVAS_MULTIPLE
        width, height = image.size
        short_edge = min(width, height)
        if min_short_edge <= short_edge <= max_short_edge:
            return image
        scale = (min_short_edge if short_edge < min_short_edge else max_short_edge) / short_edge
        size = (
            max(multiple, round(width * scale / multiple) * multiple),
            max(multiple, round(height * scale / multiple) * multiple),
        )
        if image.size == size:
            return image
        return image.resize(size, Image.LANCZOS)

    def build_reference_tokens(self, pipe: MiniMaxH3ImagePipeline, images, max_sequence_length):
        tokenizer = pipe.tokenizer
        vision = pipe.processor.image_processor(images=images, return_tensors="pt")
        merge_size = int(pipe.processor.image_processor.merge_size) ** 2
        vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        token_ids, token_tags = [], []
        for index, grid in enumerate(vision["image_grid_thw"]):
            label_ids = tokenizer(f"Picture {index + 1}: ", add_special_tokens=False)["input_ids"][:max_sequence_length]
            vision_ids = [vision_start] + [image_pad] * (int(grid.prod()) // merge_size) + [vision_end]
            token_ids += label_ids + vision_ids
            token_tags += [MINIMAX_H3_IMAGE_TEXT_TAG] * len(label_ids) + [MINIMAX_H3_IMAGE_IMAGE_TAG] * len(vision_ids)
        return token_ids, token_tags, vision["pixel_values"], vision["image_grid_thw"]

    def process(self, pipe: MiniMaxH3ImagePipeline, prompt, images, max_sequence_length):
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError(f"MiniMax-H3-Image packs one request into one sequence, so prompt must be a single non-empty string, but got {prompt!r}.")
        if images is not None:
            images = images if isinstance(images, list) else [images]
            images = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
            for image in images:
                if min(image.size) < MINIMAX_H3_IMAGE_CANVAS_MULTIPLE:
                    raise ValueError(f"A reference image needs at least {MINIMAX_H3_IMAGE_CANVAS_MULTIPLE} pixels on each side, but got {image.size}.")
            reference_images = [self.resize_reference_image(image, MINIMAX_H3_IMAGE_REFERENCE_MIN_SHORT_EDGE, MINIMAX_H3_IMAGE_REFERENCE_MAX_SHORT_EDGE) for image in images]
            vision_images = [self.resize_reference_image(image, MINIMAX_H3_IMAGE_VISION_MIN_SHORT_EDGE, MINIMAX_H3_IMAGE_VISION_MAX_SHORT_EDGE) for image in images]
            token_ids, token_tags, pixel_values, image_grid_thw = self.build_reference_tokens(pipe, vision_images, max_sequence_length)
            prompt_ids = pipe.tokenizer(prompt, add_special_tokens=False)["input_ids"][:max_sequence_length]
            token_ids += prompt_ids
            token_tags += [MINIMAX_H3_IMAGE_TEXT_TAG] * len(prompt_ids)
        else:
            reference_images = None
            pixel_values, image_grid_thw = None, None
            token_ids = pipe.tokenizer(prompt, add_special_tokens=False)["input_ids"][:max_sequence_length]
            token_tags = [MINIMAX_H3_IMAGE_TEXT_TAG] * len(token_ids)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=pipe.device)
        attention_mask = torch.ones_like(input_ids)
        prompt_embeds = pipe.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=None if pixel_values is None else pixel_values.to(pipe.device, pipe.text_encoder.model.dtype),
            image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(pipe.device),
        )
        return {
            "prompt_embeds": prompt_embeds.to(pipe.torch_dtype),
            "text_token_tags": torch.tensor(token_tags, dtype=torch.long),
            "reference_images": reference_images,
        }


class MiniMaxH3ImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("latent_height", "latent_width", "seed", "rand_device", "latents"),
            output_params=("latents", "generator"),
        )

    def process(self, pipe: MiniMaxH3ImagePipeline, latent_height, latent_width, seed, rand_device, latents=None):
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        shape = (1, int(pipe.vae.latents_mean.shape[1]), 1, latent_height, latent_width)
        if latents is None:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=torch.float32)
        else:
            latents = latents.to(torch.float32).reshape(shape)
        return {"latents": latents.to(pipe.device), "generator": generator}


class MiniMaxH3ImageUnit_ReferenceEncoder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_images", "generator", "reference_noise", "tiling"),
            output_params=("reference_latents", "reference_grids", "num_reference_rows"),
            onload_model_names=("vae",),
        )

    @staticmethod
    def sample_posterior(posterior, generator=None):
        # One sample of a diagonal Gaussian, drawn on the generator's own device: the generator lives on `rand_device`
        # (CPU by default) so that a seed gives the same result on every machine, while the posterior lives wherever
        # the VAE computes.
        device = generator.device if generator is not None else posterior.mean.device
        noise = torch.randn(posterior.mean.shape, generator=generator, device=device, dtype=posterior.mean.dtype)
        return posterior.mean + posterior.std * noise.to(posterior.mean.device)

    @torch.no_grad()
    def encode_reference_image(self, pipe: MiniMaxH3ImagePipeline, image, generator=None, tiling=True):
        # The released recipe draws one posterior sample off the request's generator, rounds it to float16 and crops
        # the result to whole patches; the latents are then normalized by `latents_mean` / `latents_std`.
        ratio = pipe.vae.spatial_compression_ratio
        _, patch_h, patch_w = MINIMAX_H3_IMAGE_PATCH_SIZE
        pixel_mean = torch.tensor(MINIMAX_H3_IMAGE_PIXEL_MEAN, device=pipe.device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_IMAGE_PIXEL_STD, device=pipe.device).view(1, -1, 1, 1)
        latents_mean = pipe.vae.latents_mean.to(pipe.device, torch.float32).view(1, -1, 1, 1)
        latents_std = pipe.vae.latents_std.to(pipe.device, torch.float32).view(1, -1, 1, 1)

        pixels = torch.from_numpy(np.array(image)).to(pipe.device).permute(2, 0, 1)[None]
        pixels = pixels[..., : pixels.shape[-2] // ratio * ratio, : pixels.shape[-1] // ratio * ratio]
        pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        posterior = pipe.vae.encode(pixels, tiling=tiling)
        latents = self.sample_posterior(posterior, generator).to(torch.float16).float()
        latents = latents[..., : latents.shape[-2] // patch_h * patch_h, : latents.shape[-1] // patch_w * patch_w]
        return (latents - latents_mean) / latents_std

    def process(self, pipe: MiniMaxH3ImagePipeline, reference_images, generator, reference_noise=None, tiling=True):
        if reference_images is None:
            return {"reference_latents": None, "reference_grids": [], "num_reference_rows": 0}
        reference_latents, reference_grids = [], []
        for index, image in enumerate(reference_images):
            latent = self.encode_reference_image(pipe, image, generator=generator, tiling=tiling)
            if reference_noise is not None:
                noise = reference_noise[index].to(pipe.device, torch.float32).reshape(latent.shape)
            else:
                noise = torch.randn(latent.shape, generator=generator, device="cpu", dtype=torch.float32).to(pipe.device)
            # The references are not fully clean: they are noised to `t = 0.999` and held there for every step. The
            # level goes in as a float32 tensor so that `1 - t` is evaluated in float32, as the released model mixes it.
            timestep = torch.tensor(MINIMAX_H3_IMAGE_REFERENCE_NOISE_AUG, dtype=torch.float32, device=pipe.device)
            noised = (timestep * latent + (1.0 - timestep) * noise)[:, :, None]
            reference_latents.append(noised)
            reference_grids.append(tuple(noised.shape[-2:]))
        num_reference_rows = sum(
            (height // MINIMAX_H3_IMAGE_PATCH_SIZE[1]) * (width // MINIMAX_H3_IMAGE_PATCH_SIZE[2])
            for height, width in reference_grids
        )
        return {"reference_latents": reference_latents, "reference_grids": reference_grids, "num_reference_rows": num_reference_rows}


class MiniMaxH3ImageUnit_InputImageEmbedder(PipelineUnit):
    """Encodes the training image into the clean latents the flow-match loss noises."""

    def __init__(self):
        super().__init__(
            input_params=("input_image", "tiling"),
            output_params=("input_latents",),
            onload_model_names=("vae",),
        )

    @torch.no_grad()
    def process(self, pipe: MiniMaxH3ImagePipeline, input_image, tiling=True):
        if input_image is None:
            return {}
        ratio = pipe.vae.spatial_compression_ratio
        pixel_mean = torch.tensor(MINIMAX_H3_IMAGE_PIXEL_MEAN, device=pipe.device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_IMAGE_PIXEL_STD, device=pipe.device).view(1, -1, 1, 1)
        latents_mean = pipe.vae.latents_mean.to(pipe.device, torch.float32).view(1, -1, 1, 1)
        latents_std = pipe.vae.latents_std.to(pipe.device, torch.float32).view(1, -1, 1, 1)
        pixels = torch.from_numpy(np.array(input_image)).to(pipe.device).permute(2, 0, 1)[None]
        pixels = pixels[..., : pixels.shape[-2] // ratio * ratio, : pixels.shape[-1] // ratio * ratio]
        pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        latents = pipe.vae.encode(pixels, tiling=tiling).mode()
        latents = (latents - latents_mean) / latents_std
        return {"input_latents": latents[:, :, None]}


class MiniMaxH3ImageUnit_TimestepPlanner(PipelineUnit):
    """Plans the noise level of every row at every sampling step.

    Step `i` runs the transformer's output head `i`, so the generated rows follow the fixed six-step grid, while the
    reference rows of an edit stay at the conditioning level `0.999` for the whole loop and every step sees the same
    pair of distinct timesteps.
    """

    def __init__(self):
        super().__init__(
            input_params=("position_ids", "image_indices", "num_reference_rows"),
            output_params=("row_timestep_plan",),
        )

    def process(self, pipe: MiniMaxH3ImagePipeline, position_ids, image_indices, num_reference_rows):
        sequence_length = int(position_ids.shape[1])
        row_timestep_plan = []
        for step_id in range(len(pipe.scheduler.timesteps)):
            row_timesteps = torch.full((sequence_length,), float(1.0 - pipe.scheduler.sigmas[step_id]), dtype=torch.float32, device=pipe.device)
            if num_reference_rows > 0:
                row_timesteps[image_indices[:num_reference_rows]] = MINIMAX_H3_IMAGE_REFERENCE_NOISE_AUG
            row_timestep_plan.append(torch.unique(row_timesteps, sorted=True, return_inverse=True))
        return {"row_timestep_plan": row_timestep_plan}


class MiniMaxH3ImageUnit_PackedSequenceBuilder(PipelineUnit):
    """Lays the text rows, the reference-image rows and the generated rows into one packed sequence.

    The rotary grid is built in float64 with numpy, which is what reproduces the released coordinates exactly.
    """

    def __init__(self):
        super().__init__(
            input_params=("text_token_tags", "reference_grids", "latent_height", "latent_width"),
            output_params=("position_ids", "token_tags", "image_indices", "text_indices", "num_reference_rows"),
        )

    @classmethod
    def spatial_position_grid(cls, dim, patch, sqrt_area):
        # One aspect-normalized spatial rotary axis: `dim // patch` coordinates centered on the unit interval, scaled
        # by 32. The right endpoint is excluded. Built with numpy because `np.linspace(..., endpoint=False)` is not
        # what `torch.linspace` computes, and the float64 grid has to be reproduced exactly.
        ratio = dim / sqrt_area
        left = (1.0 - ratio) / 2.0
        grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * MINIMAX_H3_IMAGE_ROPE_SPATIAL_SCALE
        return torch.from_numpy(grid).to(torch.float64)

    @classmethod
    def frame_position_grid(cls, latent_height, latent_width, patch_h, patch_w):
        # The (h, w) rotary coordinates of one latent frame, flattened row-major.
        sqrt_area = np.sqrt(latent_height * latent_width)
        height_grid = cls.spatial_position_grid(latent_height, patch_h, sqrt_area)
        width_grid = cls.spatial_position_grid(latent_width, patch_w, sqrt_area)
        grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
        return torch.stack([grid.reshape(-1) for grid in grids], dim=-1)

    def build_packed_sequence(self, text_token_tags, reference_grids, latent_height, latent_width):
        # The layout is `[text | reference images | generated image]`; every image is one frame on the rotary clock,
        # which starts where the text rows end: a reference takes one integer slot each, and the generated image takes
        # the slot after the last reference.
        _, patch_h, patch_w = MINIMAX_H3_IMAGE_PATCH_SIZE
        num_text_tokens = text_token_tags.shape[0]
        num_reference_rows = sum((height // patch_h) * (width // patch_w) for height, width in reference_grids)
        num_target_rows = (latent_height // patch_h) * (latent_width // patch_w)
        sequence_length = num_text_tokens + num_reference_rows + num_target_rows

        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
        position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
        cursor = num_text_tokens
        rotary_time = float(num_text_tokens)
        for height, width in reference_grids:
            rows = slice(cursor, cursor + (height // patch_h) * (width // patch_w))
            cursor = rows.stop
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = self.frame_position_grid(height, width, patch_h, patch_w)
            rotary_time += 1.0
        position_ids[cursor:, 0] = rotary_time
        position_ids[cursor:, 1:] = self.frame_position_grid(latent_height, latent_width, patch_h, patch_w)

        text_indices = torch.arange(num_text_tokens)
        image_indices = torch.arange(num_text_tokens, sequence_length)
        token_tags = torch.empty(sequence_length, dtype=torch.long)
        token_tags[text_indices] = text_token_tags.to(torch.long)
        token_tags[image_indices] = MINIMAX_H3_IMAGE_IMAGE_TAG
        return position_ids, token_tags, image_indices, text_indices, num_reference_rows

    def process(self, pipe: MiniMaxH3ImagePipeline, text_token_tags, reference_grids, latent_height, latent_width):
        position_ids, token_tags, image_indices, text_indices, num_reference_rows = self.build_packed_sequence(
            text_token_tags, reference_grids, latent_height, latent_width,
        )
        return {
            "position_ids": position_ids.to(pipe.device).unsqueeze(0),
            "token_tags": token_tags.to(pipe.device),
            "image_indices": image_indices.to(pipe.device),
            "text_indices": text_indices.to(pipe.device),
            "num_reference_rows": num_reference_rows,
        }


def model_fn_minimax_h3_image(
    dit,
    pdd_heads,
    latents,
    prompt_embeds,
    position_ids,
    token_tags,
    image_indices,
    text_indices,
    reference_latents=None,
    num_reference_rows=0,
    timestep=None,
    unique_timesteps=None,
    timestep_indices=None,
    pdd_step_index=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    """Predicts the velocity of the generated rows of the packed sequence.

    The transformer emits a data-ward velocity (`x0 = x_t + sigma * v`) for every image row, reference rows included;
    the generated rows are selected and the sign is flipped, because the rest of DiffSynth (the scheduler step and the
    flow-match training target `noise - x0`) works with the noise-ward velocity.
    """
    latent_height, latent_width = int(latents.shape[-2]), int(latents.shape[-1])
    rows = [patchify_video(reference_latent) for reference_latent in (reference_latents or [])]
    rows.append(patchify_video(latents))
    rows = torch.cat(rows) if len(rows) > 1 else rows[0]

    if unique_timesteps is None:
        # Training: every row of the sequence sits at the timestep the loss sampled, and the output head of that noise
        # level is the one the checkpoint distilled for it.
        unique_timesteps = (1.0 - timestep.float().view(-1) / 1000).to(device=rows.device)
        timestep_indices = torch.zeros(position_ids.shape[1], dtype=torch.long, device=rows.device)
    if pdd_step_index is None:
        # Training: the output head of the sampled noise level is the one the checkpoint distilled for it.
        sample_sigmas = torch.tensor(MINIMAX_H3_IMAGE_SAMPLE_SIGMAS, dtype=torch.float32, device=unique_timesteps.device)
        pdd_step_index = int(torch.argmin((sample_sigmas - unique_timesteps[0].float()).abs()))

    velocity_rows = dit(
        x=rows,
        prompt_embeds=prompt_embeds,
        position_ids=position_ids,
        unique_timesteps=unique_timesteps,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        img_pos=image_indices,
        text_pos=text_indices,
        pdd_heads=pdd_heads,
        pdd_step_index=pdd_step_index,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    velocity_rows = velocity_rows[num_reference_rows:]
    velocity = unpatchify_video(velocity_rows, 1, latent_height, latent_width, channel=MINIMAX_H3_IMAGE_LATENT_CHANNELS)
    return -velocity
