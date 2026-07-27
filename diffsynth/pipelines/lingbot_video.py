import json
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, Union
from typing_extensions import Literal

from ..core.device.npu_compatible_device import get_device_type
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..diffusion.flow_match import FlowMatchScheduler

from ..models.lingbot_video_dit import LingBotVideoDiT
from ..models.lingbot_video_text_encoder import LingBotVideoTextEncoder
from ..models.qwen_image_vae import QwenImageVAE, QwenImageCausalConv3d


# LingBot-Video is trained on structured JSON captions. normalize_caption serialises a
# dict/list caption (or a prompt.json path) into the compact-JSON string the DiT expects;
# a plain string is passed through. Kept module-level so train.py / rewrite_captions.py
# can reuse it without importing the pipeline. The prompt rewriter that turns a brief idea
# into such a caption lives in examples/lingbot_video/model_inference/prompt_rewriter.py.
_RUNTIME_KEYS = {"duration", "fps", "height", "width", "num_frames", "resolution", "ratio"}


def _serialize_caption(caption) -> str:
    if isinstance(caption, (dict, list)):
        return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
    return str(caption)


def _caption_from_sample(sample) -> str:
    if isinstance(sample, dict):
        if "caption" in sample:
            caption = sample["caption"]
        else:
            caption = {k: v for k, v in sample.items() if k not in _RUNTIME_KEYS}
    else:
        caption = sample
    return _serialize_caption(caption)


def normalize_caption(prompt):
    if prompt is None:
        return prompt
    if isinstance(prompt, str):
        if prompt.endswith(".json") and os.path.isfile(prompt):
            with open(prompt, "r", encoding="utf-8") as f:
                prompt = json.load(f)
            return _caption_from_sample(prompt)
        return prompt
    if isinstance(prompt, (dict, list)):
        return _caption_from_sample(prompt)
    return str(prompt)


# Prompt truncation length for the Qwen3-VL processor.
TOKEN_LENGTH = 37698
# Hidden-state layer used as the prompt embedding: 0 -> the last layer.
HIDDEN_STATE_SKIP_LAYER = 0

# Prompt-enhancement chat template wrapping the user prompt.
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

# Default T2V negative prompt (structured JSON string), copied verbatim.
DEFAULT_NEGATIVE_PROMPT = (
    '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'
)

# VAE downsample factors (QwenImageVAE / Wan-VAE): 8x spatial, 4x temporal.
VAE_SCALE_FACTOR_SPATIAL = 8
VAE_SCALE_FACTOR_TEMPORAL = 4


class LingBotVideoPipeline(BasePipeline):
    """Text-to-video pipeline for LingBot-Video (DiT + Qwen3-VL text encoder + QwenImageVAE),
    following the DiffSynth PipelineUnit + model_fn pattern. Sampling uses FlowMatchScheduler
    (Wan template). The 5D-video VAE encode/decode lives here (encode_video / decode_video)
    so QwenImageVAE stays identical to its image use elsewhere."""

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
            time_division_factor=4, time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler(template="Wan")
        self.text_encoder: LingBotVideoTextEncoder = None
        self.dit: LingBotVideoDiT = None
        self.vae: QwenImageVAE = None
        self.processor = None
        # Cached number of template-prefix tokens to crop from the prompt embedding.
        self._crop_start: Optional[int] = None
        self.in_iteration_models = ("dit",)
        self.units = [
            LingBotVideoUnit_ShapeChecker(),
            LingBotVideoUnit_NoiseInitializer(),
            LingBotVideoUnit_PromptEmbedder(),
            LingBotVideoUnit_InputVideoEmbedder(),
        ]
        self.model_fn = model_fn_lingbot_video
        self.compilable_models = ["dit"]

    def _compute_crop_start(self) -> int:
        # Token count of the template prefix (everything before the user prompt), computed once.
        if self._crop_start is None:
            marker = "<|USER_INPUT_MARKER|>"
            marked = PROMPT_TEMPLATE.format(marker)
            marker_pos = marked.find(marker)
            if marker_pos < 0:
                self._crop_start = 0
            else:
                prefix = self.processor(
                    text=marked[:marker_pos],
                    images=None,
                    videos=None,
                    return_tensors="pt",
                )
                self._crop_start = int(prefix["input_ids"].shape[1])
        return self._crop_start

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = LingBotVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models by name (registered in diffsynth/configs/model_configs.py).
        pipe.text_encoder = model_pool.fetch_model("lingbot_video_text_encoder")
        pipe.dit = model_pool.fetch_model("lingbot_video_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_vae")

        # Initialize the Qwen3-VL processor (tokenizer + image/video processor).
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
        # Structured caption (dict / list), a prompt.json path, or a plain string.
        prompt: Union[str, dict, list] = "",
        negative_prompt: Union[str, dict, list] = DEFAULT_NEGATIVE_PROMPT,
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
        cfg_scale: float = 6.0,
        # Scheduler
        num_inference_steps: int = 40,
        sigma_shift: float = 3.0,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Serialise dict/list/prompt.json captions to the structured-JSON string; plain
        # strings pass through unchanged.
        prompt = normalize_caption(prompt)
        negative_prompt = normalize_caption(negative_prompt)

        # Inputs
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
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
            # fp32 so the integer timestep is represented exactly (bf16 rounds values > 256).
            timestep_input = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)

            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep_input)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep_input)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            inputs_shared["latents"] = self.scheduler.step(noise_pred, timestep, inputs_shared["latents"])

        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"].to(dtype=self.torch_dtype, device=self.device)
        video = self.decode_video(latents)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video

    def _count_conv3d(self, model):
        return sum(1 for m in model.modules() if isinstance(m, QwenImageCausalConv3d))

    def encode_video(self, x):
        # x: (B, C, T, H, W). Temporal chunking (1 + 4k frames) through a persistent causal
        # feature cache — equivalent to encoding the whole clip at once, bounded in memory.
        vae = self.vae
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        feat_cache = [None] * self._count_conv3d(vae.encoder)
        out = None
        for i in range(iter_):
            feat_idx = [0]
            chunk = x[:, :, :1, :, :] if i == 0 else x[:, :, 1 + 4 * (i - 1): 1 + 4 * i, :, :]
            out_ = vae.encoder(chunk, feat_cache=feat_cache, feat_idx=feat_idx)
            out = out_ if out is None else torch.cat([out, out_], dim=2)
        x = vae.quant_conv(out)
        x = x[:, :16]
        mean, std = vae.mean.to(dtype=x.dtype, device=x.device), vae.std.to(dtype=x.dtype, device=x.device)
        x = (x - mean) * std
        return x

    def decode_video(self, x):
        # x: (B, 16, T', H, W) in latent space. Denormalize, then decode one latent frame
        # at a time through the causal feature cache.
        vae = self.vae
        mean, std = vae.mean.to(dtype=x.dtype, device=x.device), vae.std.to(dtype=x.dtype, device=x.device)
        x = x / std + mean
        x = vae.post_quant_conv(x)
        feat_cache = [None] * self._count_conv3d(vae.decoder)
        out = None
        for i in range(x.shape[2]):
            feat_idx = [0]
            out_ = vae.decoder(x[:, :, i:i + 1, :, :], feat_cache=feat_cache, feat_idx=feat_idx)
            out = out_ if out is None else torch.cat([out, out_], dim=2)
        return out


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
        length = (num_frames - 1) // VAE_SCALE_FACTOR_TEMPORAL + 1
        shape = (
            1, pipe.dit.in_channels, length,
            height // VAE_SCALE_FACTOR_SPATIAL, width // VAE_SCALE_FACTOR_SPATIAL,
        )
        # fp32 noise: the flow-matching sampler accumulates state in fp32.
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device, torch_dtype=torch.float32)
        return {"noise": noise}


class LingBotVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("context", "encoder_attention_mask"),
            onload_model_names=("text_encoder",),
        )

    def encode_prompt(self, pipe: LingBotVideoPipeline, prompt):
        # T2V: visual_template is empty, so the text is simply the templated prompt.
        text = PROMPT_TEMPLATE.format(prompt)
        inputs = pipe.processor(
            text=[text],
            images=None,
            videos=None,
            do_resize=False,
            truncation=True,
            max_length=TOKEN_LENGTH,
            padding="longest",
            return_tensors="pt",
        )
        inputs = inputs.to(pipe.device)
        # The text encoder returns the tuple of per-layer hidden states.
        hidden_states = pipe.text_encoder(**inputs)
        prompt_embeds = hidden_states[-(HIDDEN_STATE_SKIP_LAYER + 1)]
        prompt_mask = inputs["attention_mask"]

        # Crop the prompt-enhancement template prefix.
        crop_start = pipe._compute_crop_start()
        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_mask = prompt_mask[:, crop_start:]

        # Batch=1: drop the right padding before the DiT forward.
        if prompt_embeds.shape[0] == 1:
            true_len = int(prompt_mask[0].sum().item())
            prompt_embeds = prompt_embeds[:, :true_len]
            prompt_mask = prompt_mask[:, :true_len]

        return prompt_embeds.to(dtype=pipe.torch_dtype), prompt_mask

    def process(self, pipe: LingBotVideoPipeline, prompt) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds, prompt_mask = self.encode_prompt(pipe, prompt)
        return {"context": prompt_embeds, "encoder_attention_mask": prompt_mask}


class LingBotVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: LingBotVideoPipeline, input_video, noise):
        if input_video is None:
            # Text-to-video: start from pure noise.
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        # preprocess_video -> (B, C, T, H, W) in [-1, 1], in pipe.torch_dtype.
        video = pipe.preprocess_video(input_video)
        # encode_video applies latent normalisation on the 5D-video path.
        input_latents = pipe.encode_video(video).to(dtype=torch.float32, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


def model_fn_lingbot_video(
    dit: LingBotVideoDiT,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    # Cast inputs to the DiT's compute dtype (e.g. bf16); the DiT keeps AdaLN/norm in fp32.
    dit_dtype = dit.patch_embedder.weight.dtype
    hidden_states = latents.to(dtype=dit_dtype)
    encoder_hidden_states = context.to(dtype=dit_dtype)
    noise_pred = dit(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return noise_pred.float()
