import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, Union
from typing_extensions import Literal

from ..core.device.npu_compatible_device import get_device_type
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..diffusion.flow_match import LingBotVideoUniPCScheduler

from ..models.lingbot_video_dit import LingBotVideoDiT
from ..models.lingbot_video_text_encoder import LingBotVideoTextEncoder
from ..models.qwen_image_vae import QwenImageVAE


# Number of tokens the Qwen3-VL processor truncates the prompt to. Copied verbatim
# from the original lingbot-video pipeline so the encoded prompt matches.
TOKEN_LENGTH = 37698
# Which hidden-state layer to use as the prompt embedding: 0 -> the last layer.
HIDDEN_STATE_SKIP_LAYER = 0

# Chat template that wraps the user prompt inside the prompt-enhancement system
# prompt. `apply_text_to_template(prompt) == PROMPT_TEMPLATE.format(prompt)`.
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
    """
    Text-to-video pipeline for LingBot-Video.

    Follows the DiffSynth ``PipelineUnit`` + ``model_fn`` pattern (see
    :class:`~diffsynth.pipelines.wan_video.WanVideoPipeline`). Components:

    - ``dit``: :class:`~diffsynth.models.lingbot_video_dit.LingBotVideoDiT` (MoE /
      Dense video DiT), conditioned on ``timestep`` and ``encoder_attention_mask``.
    - ``text_encoder``:
      :class:`~diffsynth.models.lingbot_video_text_encoder.LingBotVideoTextEncoder`
      (Qwen3-VL). The prompt is wrapped in :data:`PROMPT_TEMPLATE`, encoded, and the
      template-prefix tokens are cropped (``crop_start``).
    - ``vae``: :class:`~diffsynth.models.qwen_image_vae.QwenImageVAE` (byte-identical
      to the LingBot-Video VAE). Latent normalisation is baked into the VAE's
      ``encode``/``decode`` 5D-video code path, so the pipeline never re-applies
      ``latents_mean`` / ``latents_std``.

    Sampling uses :class:`~diffsynth.diffusion.flow_match.LingBotVideoUniPCScheduler`
    (UniPC multistep). Classifier-free guidance runs as two independent forwards.
    """

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
            time_division_factor=4, time_division_remainder=1,
        )
        self.scheduler = LingBotVideoUniPCScheduler()
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
        # Number of tokens contributed by the template prefix (everything before the
        # user prompt). Computed once by tokenising the template up to a marker.
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
        # Prompt
        prompt: str = "",
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
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
            # The DiT is conditioned on sigma * 1000 (== the raw scheduler timestep).
            # Pass it as fp32 so the integer inference timesteps are represented
            # exactly (bf16 cannot represent values > 256 without rounding).
            timestep_input = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)

            # Inference (two independent forwards for CFG).
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep_input)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep_input)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler step (UniPC multistep). Uses the raw scheduler timestep to
            # locate its internal step index, so pass it unmodified.
            inputs_shared["latents"] = self.scheduler.step(noise_pred, timestep, inputs_shared["latents"])

        # Decode. The VAE's 5D-video path already un-normalises the latents, so no
        # manual latents_mean / latents_std handling is needed here.
        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"].to(dtype=self.torch_dtype, device=self.device)
        video = self.vae.decode(latents)
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
        length = (num_frames - 1) // VAE_SCALE_FACTOR_TEMPORAL + 1
        shape = (
            1, pipe.dit.in_channels, length,
            height // VAE_SCALE_FACTOR_SPATIAL, width // VAE_SCALE_FACTOR_SPATIAL,
        )
        # fp32 noise: the UniPC sampler accumulates state in fp32 for stability
        # (matches the original pipeline's fp32 latents).
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
        # QwenImageVAE.encode (5D path) already applies latent normalisation.
        input_latents = pipe.vae.encode(video).to(dtype=torch.float32, device=pipe.device)
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
    # Cast the latent / text inputs to the DiT's bulk compute dtype (e.g. bf16).
    # The DiT keeps its AdaLN / norm / router paths in fp32 internally.
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
    # Return fp32 so the UniPC sampler / MSE loss run in full precision.
    return noise_pred.float()
