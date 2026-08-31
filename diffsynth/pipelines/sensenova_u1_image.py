# SenseNova-U1 Image Pipeline for DiffSynth-Studio.

import torch
from typing import Union
from PIL import Image
from tqdm import tqdm

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import SenseNovaU1Scheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.sensenova_u1_dit import SenseNovaU1DiT
from ..models.sensenova_u1_common import (
    IMAGE_PLACEHOLDER, IMG_CONTEXT_TOKEN, IMG_START_TOKEN, NON_THINK_PREFIX, PATCH_SIZE,
    SYSTEM_MESSAGE_FOR_GEN, THINK_PREFIX, build_conversation_prompt, build_image_token_block,
    build_thw_indexes, create_block_causal_mask, load_image_native, patchify, unpatchify,
)


class SenseNovaU1ImagePipeline(BasePipeline):
    """Pipeline for SenseNova-U1 unified multimodal image generation.

    Flow matching runs directly in pixel space, so there is no VAE: `latents` holds a
    (1, 3, H, W) image tensor throughout and the final result is returned by
    `vae_output_to_image` without a decode step.

    Conditioning goes through a KV cache rather than a separate text encoder. The DiT's
    understanding branch encodes the prompt once into `past_key_values`, and every denoising
    step runs the image tokens through the generation branch against that cache. The two
    branches share the same weights, which is why they cannot be split into separate models.

    Passing `edit_image` switches to the editing task: the input images are encoded by the
    understanding-branch vision encoder and spliced into that same prefix, and the negative
    branch carries the images without an instruction. Image guidance weighted separately from
    text guidance is not supported, since it needs a third prefix cache.

    There is no `negative_prompt`. The unconditional prefix is fixed, as it is in the reference.

    Setting `think_mode` lets the model write a reasoning block before generating, which is decoded
    into the conditioning cache. The reasoning shapes the image but is not returned.
    """

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=PATCH_SIZE, width_division_factor=PATCH_SIZE,
        )
        self.scheduler = SenseNovaU1Scheduler()
        self.dit: SenseNovaU1DiT = None
        self.tokenizer = None

        self.in_iteration_models = ("dit",)
        self.units = [
            SenseNovaU1ImageUnit_ShapeChecker(),
            SenseNovaU1ImageUnit_EditImageEmbedder(),
            SenseNovaU1ImageUnit_PromptEmbedder(),
            SenseNovaU1ImageUnit_NoiseInitializer(),
            SenseNovaU1ImageUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_sensenova_u1_image
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="./"),
        vram_limit: float = None,
    ):
        pipe = SenseNovaU1ImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.dit = model_pool.fetch_model("sensenova_u1_dit")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        cfg_scale: float = 4.0,
        # Shape
        height: int = 2048,
        width: int = 2048,
        # Randomness
        seed: int = None,
        rand_device: str = "cuda",
        # Steps
        num_inference_steps: int = 50,
        # Scheduler
        shift: float = 3.0,
        # Reasoning
        think_mode: bool = False,
        # Image editing
        edit_image: Union[Image.Image, list[Image.Image]] = None,
        # Progress bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, shift=shift)

        # Parameters
        inputs_posi = {"prompt": prompt, "prompt_is_negative": False}
        inputs_nega = {"negative_is_negative": True}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "think_mode": think_mode,
            "edit_image": edit_image,
        }

        # Units
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.to(dtype=torch.float32, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

        image = self.vae_output_to_image(inputs_shared["latents"])
        self.load_models_to_device([])
        return image


class SenseNovaU1ImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: SenseNovaU1ImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class SenseNovaU1ImageUnit_EditImageEmbedder(PipelineUnit):
    """Preprocess and patchify the editing input images.

    The per-image pixel budget shrinks as the image count grows. This runs once in the shared
    dict because the conditional and image-conditional prefixes are built from the same tensor.
    """

    def __init__(self):
        super().__init__(
            input_params=("edit_image",),
            output_params=("edit_pixel_values", "edit_grid_hw"),
        )

    def process(self, pipe: SenseNovaU1ImagePipeline, edit_image):
        if edit_image is None:
            return {"edit_pixel_values": None, "edit_grid_hw": None}
        if isinstance(edit_image, Image.Image):
            edit_image = [edit_image]

        max_pixels = min(2048 * 2048, (4096 * 4096) // len(edit_image))
        pixel_values, grid_hw = [], []
        for image in edit_image:
            cur_pixel_values, cur_grid_hw = load_image_native(
                image, pipe.dit.patch_size, pipe.dit.downsample_ratio,
                min_pixels=512 * 512, max_pixels=max_pixels, upscale=False,
            )
            pixel_values.append(cur_pixel_values.to(device=pipe.device, dtype=pipe.torch_dtype))
            grid_hw.append(cur_grid_hw.to(pipe.device))
        return {"edit_pixel_values": torch.cat(pixel_values), "edit_grid_hw": torch.cat(grid_hw)}


class SenseNovaU1ImageUnit_PromptEmbedder(PipelineUnit):
    """Encode the conditioning prefix into a KV cache via the understanding branch.

    The conditional and unconditional branches are asymmetric: the conditional side carries the
    generation system message plus an empty reasoning block, while the unconditional side uses an
    empty system message and an empty prompt.

    With editing inputs the unconditional branch changes meaning: instead of an empty prompt it
    carries the input images with no instruction, so guidance pushes away from "the input image
    unchanged" rather than from "any image".

    Think mode leaves the reasoning block open so the model writes it itself, then greedily decodes
    it into the same cache. Only the conditional side reasons; the unconditional side is unchanged.
    Because the cache grows, the image tokens that follow start after the reasoning block rather
    than after the original prompt.
    """

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=("height", "width", "edit_pixel_values", "edit_grid_hw", "think_mode"),
            input_params_posi={"prompt": "prompt", "is_negative": "prompt_is_negative"},
            input_params_nega={"is_negative": "negative_is_negative"},
            output_params=("past_key_values", "indexes_image"),
            onload_model_names=("dit",),
        )

    @staticmethod
    def insert_image_placeholders(prompt, num_images):
        """Prepend `<image>` placeholders for every input image the prompt does not mention."""
        missing = num_images - prompt.count(IMAGE_PLACEHOLDER)
        if missing <= 0:
            return prompt
        if missing == num_images and num_images > 1:
            return "".join(f"Image-{i + 1}:{IMAGE_PLACEHOLDER}\n" for i in range(num_images)) + prompt
        return f"{IMAGE_PLACEHOLDER}\n" * missing + prompt

    @staticmethod
    def expand_image_placeholders(query, grid_hw, downsample_ratio):
        """Replace each `<image>` placeholder with that image's run of context tokens."""
        for i in range(grid_hw.shape[0]):
            num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * downsample_ratio ** 2)
            query = query.replace(IMAGE_PLACEHOLDER, build_image_token_block(num_patch_token), 1)
        return query

    @staticmethod
    def build_prefix_inputs(pipe, query, pixel_values, grid_hw):
        input_ids = pipe.tokenizer(query, return_tensors="pt")["input_ids"].to(pipe.device)
        indexes = build_thw_indexes(
            input_ids[0],
            pipe.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN),
            pipe.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN),
            grid_hw=grid_hw,
            merge_size=pipe.dit.merge_size,
        )
        attention_mask = create_block_causal_mask(indexes[0])
        if pixel_values is None:
            return input_ids, None, indexes, attention_mask

        inputs_embeds = pipe.dit.embed_tokens(input_ids)
        batch_size, num_tokens, channels = inputs_embeds.shape
        vision_embeds = pipe.dit.extract_und_feature(pixel_values, grid_hw)
        selected = input_ids.reshape(-1) == pipe.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        inputs_embeds = inputs_embeds.reshape(-1, channels)
        inputs_embeds[selected] = vision_embeds.reshape(-1, channels).to(inputs_embeds.device)
        return input_ids, inputs_embeds.reshape(batch_size, num_tokens, channels), indexes, attention_mask

    @staticmethod
    def build_image_indexes(token_h, token_w, text_len, device):
        t_image = torch.full((token_h * token_w,), text_len, dtype=torch.long, device=device)
        idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
        h_image = idx // token_w
        w_image = idx % token_w
        return torch.stack([t_image, h_image, w_image], dim=0)

    def process(self, pipe: SenseNovaU1ImagePipeline, is_negative, height, width, edit_pixel_values, edit_grid_hw, think_mode, prompt=None):
        pipe.load_models_to_device(self.onload_model_names)
        num_images = 0 if edit_grid_hw is None else edit_grid_hw.shape[0]
        # Reasoning is inference-only. An autoregressive decode inside a training step would grow the
        # prefix by a data-dependent length with no gradient path through the sampled tokens.
        reasoning = bool(think_mode) and not is_negative and not pipe.scheduler.training

        if is_negative:
            # The unconditional prefix carries the input images but no instruction and no system
            # message. It is fixed, matching the reference, which offers no negative prompt.
            query = build_conversation_prompt(
                IMAGE_PLACEHOLDER * num_images, system_message="", append_text=IMG_START_TOKEN,
            )
        else:
            query = build_conversation_prompt(
                self.insert_image_placeholders(prompt, num_images),
                system_message=SYSTEM_MESSAGE_FOR_GEN,
                append_text=THINK_PREFIX if reasoning else NON_THINK_PREFIX + IMG_START_TOKEN,
            )
        if num_images > 0:
            query = self.expand_image_placeholders(query, edit_grid_hw, pipe.dit.downsample_ratio)

        input_ids, inputs_embeds, indexes, attention_mask = self.build_prefix_inputs(
            pipe, query, edit_pixel_values, edit_grid_hw,
        )
        if reasoning:
            past_key_values, text_len, token_ids = pipe.dit.generate_think(
                input_ids=None if inputs_embeds is not None else input_ids,
                inputs_embeds=inputs_embeds, indexes=indexes, attention_mask=attention_mask,
                eos_token_id=pipe.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                think_end_token_id=pipe.tokenizer.convert_tokens_to_ids("</think>"),
                append_ids=pipe.tokenizer(
                    "\n\n" + IMG_START_TOKEN, return_tensors="pt", add_special_tokens=False,
                )["input_ids"].to(pipe.device),
            )
            # `token_ids` holds the reasoning the model just wrote. Nothing downstream needs it,
            # so it is left undecoded; to read it, decode it here:
            # pipe.tokenizer.decode(token_ids, skip_special_tokens=False)
        else:
            past_key_values, _ = pipe.dit.encode_prefix(
                input_ids=None if inputs_embeds is not None else input_ids,
                inputs_embeds=inputs_embeds, indexes=indexes, attention_mask=attention_mask,
            )
            text_len = indexes[0].max()

        token_h, token_w = height // PATCH_SIZE, width // PATCH_SIZE
        indexes_image = self.build_image_indexes(token_h, token_w, text_len + 1, pipe.device)
        return {"past_key_values": past_key_values, "indexes_image": indexes_image}


class SenseNovaU1ImageUnit_NoiseInitializer(PipelineUnit):
    """Initialize pixel-space noise scaled by the resolution-dependent noise scale.

    The scale is `min(sqrt(tokens / noise_scale_base_image_seq_len), noise_scale_max_value)`,
    which is not 1, so unit variance noise has to be rescaled before use.
    """

    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise", "noise_scale"),
        )

    def process(self, pipe: SenseNovaU1ImagePipeline, height, width, seed, rand_device):
        noise_scale = pipe.dit.compute_noise_scale(height, width)
        noise = pipe.generate_noise(
            (1, 3, height, width), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype
        )
        return {"noise": noise_scale * noise, "noise_scale": noise_scale}


class SenseNovaU1ImageUnit_InputImageEmbedder(PipelineUnit):
    """Provide the training target in pixel space.

    `input_image` reaches this unit only from the training module, since inference always starts
    from pure noise. There is no VAE, so the preprocessed image tensor is passed straight through
    as `input_latents`.
    """

    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
        )

    def process(self, pipe: SenseNovaU1ImagePipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        input_latents = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        return {"latents": noise, "input_latents": input_latents}


def model_fn_sensenova_u1_image(
    dit: SenseNovaU1DiT,
    latents=None,
    timestep=None,
    past_key_values=None,
    indexes_image=None,
    noise_scale=None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    """One denoising step.

    Returns `(latents - x_pred) / sigma` rather than the clean-image prediction: DiffSynth's
    scheduler steps along the noise direction, so the result is the velocity pointing from the
    predicted clean image toward the noise.
    """
    sigma = (timestep / 1000.0).flatten()[0]
    t = 1.0 - sigma

    grid_hw = torch.tensor([[latents.shape[2] // dit.patch_size, latents.shape[3] // dit.patch_size]] * latents.shape[0], device=latents.device)

    z = patchify(latents, PATCH_SIZE)
    num_tokens = z.shape[1]

    image_input = patchify(latents, dit.patch_size, channel_first=True)
    image_embeds = dit.extract_gen_feature(image_input.flatten(0, 1), grid_hw).view(latents.shape[0], num_tokens, -1)
    image_embeds = image_embeds + dit.embed_timestep(t, noise_scale, num_tokens, batch_size=latents.shape[0])

    x_pred = dit(
        image_embeds=image_embeds,
        indexes_image=indexes_image,
        past_key_values=past_key_values,
        image_size=(latents.shape[3], latents.shape[2]),
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    x_pred = unpatchify(x_pred, PATCH_SIZE, latents.shape[2], latents.shape[3])
    return (latents - x_pred) / sigma
