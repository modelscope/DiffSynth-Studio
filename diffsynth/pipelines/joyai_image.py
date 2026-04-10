import torch
from PIL import Image
from typing import Union, List
from tqdm import tqdm
from einops import rearrange
from typing import Optional, Union

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.joyai_image_dit import Transformer3DModel
from ..models.joyai_image_text_encoder import JoyAIImageTextEncoder
from ..models.joyai_image_common import _dynamic_resize_from_bucket
from ..models.wan_video_vae import WanVideoVAE

# ============================================================
# JoyAIImagePipeline
# ============================================================
class JoyAIImagePipeline(BasePipeline):
    """
    Pipeline for JoyAI-Image model.
    """

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.text_encoder: JoyAIImageTextEncoder = None
        self.dit: Transformer3DModel = None
        self.vae: WanVideoVAE = None
        self.processor = None
        self.in_iteration_models = ("dit",)

        self.units = [
            JoyAIImageUnit_ShapeChecker(),
            JoyAIImageUnit_EditImageEmbedder(),
            JoyAIImageUnit_PromptEmbedder(),
            JoyAIImageUnit_NoiseInitializer(),
            JoyAIImageUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_joyai_image
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = JoyAIImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("joyai_image_text_encoder")
        pipe.dit = model_pool.fetch_model("joyai_image_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")

        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import AutoProcessor
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    # ============================================================
    # __call__ — Orchestration only
    # ============================================================
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 5.0,
        input_image: Image.Image = None,
        edit_images: Union[Image.Image, List[Image.Image]] = None,
        edit_image_basesize: int = 1024,
        denoising_strength: float = 1.0,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        max_sequence_length: int = 4096,
        num_inference_steps: int = 30,
        tiled: Optional[bool] = False,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        shift: Optional[float] = 4.0,
        progress_bar_cmd=tqdm,
    ):
        # 1. Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=shift)

        # 2. Three dictionaries
        inputs_posi = {"prompt": prompt, "positive": True}
        inputs_nega = {"negative_prompt": negative_prompt, "positive": True}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image,
            "edit_images": edit_images, "edit_image_basesize": edit_image_basesize,
            "denoising_strength": denoising_strength,
            "height": height,
            "width": width,
            "seed": seed,
            "max_sequence_length": max_sequence_length,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
        }

        # 3. Unit chain
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
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # 5. VAE decode
        self.load_models_to_device(['vae'])
        latents = rearrange(inputs_shared["latents"], "b n c f h w -> (b n) c f h w")
        image = self.vae.decode(latents, device=self.device)[0]
        image = self.vae_output_to_image(image, pattern="C 1 H W")
        self.load_models_to_device([])
        return image


# ============================================================
# PipelineUnits
# ============================================================
class JoyAIImageUnit_ShapeChecker(PipelineUnit):
    """Validates height/width divisible by 16."""
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: "JoyAIImagePipeline", height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class JoyAIImageUnit_PromptEmbedder(PipelineUnit):
    prompt_template_encode = {
        'image':
            "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        'multiple_images':
            "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n{}<|im_start|>assistant\n",
        'video':
            "<|im_start|>system\n \\nDescribe the video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    }
    prompt_template_encode_start_idx = {'image': 34, 'multiple_images': 34, 'video': 91}
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            input_params=("edit_images", "max_sequence_length"),
            output_params=("prompt_embeds", "prompt_embeds_mask"),
            onload_model_names=("joyai_image_text_encoder",),
        )

    def process(self, pipe: "JoyAIImagePipeline", prompt, positive, edit_images, max_sequence_length):
        pipe.load_models_to_device(self.onload_model_names)

        has_image = edit_images is not None

        if has_image:
            prompt_embeds, prompt_embeds_mask = self._encode_with_image(pipe, prompt, edit_images, max_sequence_length)
        else:
            prompt_embeds, prompt_embeds_mask = self._encode_text_only(pipe, prompt, max_sequence_length)

        return {"prompt_embeds": prompt_embeds, "prompt_embeds_mask": prompt_embeds_mask}

    def _encode_with_image(self, pipe, prompt, edit_images, max_sequence_length):
        template = self.prompt_template_encode['multiple_images']
        drop_idx = self.prompt_template_encode_start_idx['multiple_images']

        image_tokens = '<image>\n'
        prompt = f"<|im_start|>user\n{image_tokens}{prompt}<|im_end|>\n"
        prompt = prompt.replace('<image>\n', '<|vision_start|><|image_pad|><|vision_end|>')
        prompt = template.format(prompt)
        inputs = pipe.processor(text=[prompt], images=edit_images, padding=True, return_tensors="pt").to(pipe.device)
        encoder_hidden_states = pipe.text_encoder(**inputs, output_hidden_states=True)
        last_hidden_states = encoder_hidden_states.hidden_states[-1]

        prompt_embeds = last_hidden_states[:, drop_idx:]
        prompt_embeds_mask = inputs['attention_mask'][:, drop_idx:]

        if max_sequence_length is not None and prompt_embeds.shape[1] > max_sequence_length:
            prompt_embeds = prompt_embeds[:, -max_sequence_length:, :]
            prompt_embeds_mask = prompt_embeds_mask[:, -max_sequence_length:]

        return prompt_embeds, prompt_embeds_mask

    def _encode_text_only(self, pipe, prompt, max_sequence_length):
        # TODO:
        template = "<think>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:</think>\n{}<think>assistant\n"
        drop_idx = 34

        txt = template.format(prompt)
        txt_tokens = pipe.processor.tokenizer(
            [txt], max_length=max_sequence_length + drop_idx,
            padding=True, truncation=True, return_tensors="pt"
        ).to(pipe.device)

        hidden_states = pipe.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )[-1]

        bool_mask = txt_tokens.attention_mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_hidden = torch.split(selected, valid_lengths.tolist(), dim=0)
        split_hidden = [e[drop_idx:] for e in split_hidden]
        attn_masks = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden]

        max_seq_len = min(max_sequence_length, max(u.size(0) for u in split_hidden))
        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden
        ])
        encoder_attention_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_masks
        ])

        return prompt_embeds, encoder_attention_mask


class JoyAIImageUnit_EditImageEmbedder(PipelineUnit):
    """
    """
    def __init__(self):
        super().__init__(
            input_params=("edit_images", "tiled", "tile_size", "tile_stride", "edit_image_basesize"),
            output_params=("ref_latents", "num_items", "is_multi_item"),
            onload_model_names=("wan_video_vae",),
        )

    def process(self, pipe: "JoyAIImagePipeline", edit_images, tiled, tile_size, tile_stride, edit_image_basesize=1024):
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(edit_images, Image.Image):
            edit_images = [edit_images]
        assert len(edit_images) == 1, "Currently only supports single edit image for reference. Multiple edit images will be supported in the future."
        edit_images = [_dynamic_resize_from_bucket(img, basesize=edit_image_basesize) for img in edit_images]

        images = [pipe.preprocess_image(img).transpose(0, 1) for img in edit_images]
        latents = pipe.vae.encode(images, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        ref_vae = rearrange(latents, "(b n) c 1 h w -> b n c 1 h w", n=(len(edit_images))).to(device=pipe.device, dtype=pipe.torch_dtype)
        
        return {"ref_latents": ref_vae, "edit_images": edit_images}


class JoyAIImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("seed", "height", "width", "rand_device"),
            output_params=("noise"),
        )
    def process(self, pipe: "JoyAIImagePipeline", seed, height, width, rand_device):
        latent_h = height // pipe.vae.upsampling_factor
        latent_w = width // pipe.vae.upsampling_factor
        shape = (1, 1, pipe.vae.z_dim, 1, latent_h, latent_w)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}


class JoyAIImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: JoyAIImagePipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise}
        raise NotImplementedError("Input image to latents is not implemented yet. Currently only supports noise initialization when input_image is None.")

# ============================================================
# model_fn — DiT forward call
# ============================================================
def model_fn_joyai_image(
    dit,
    latents,
    timestep,
    prompt_embeds,
    prompt_embeds_mask,
    ref_latents=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):

    img = torch.cat([ref_latents, latents], dim=1) if ref_latents is not None else latents

    img, _ = dit(
        hidden_states=img,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )

    img = img[:, -latents.size(1):]
    return img
