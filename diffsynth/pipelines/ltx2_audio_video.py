import torch, types
import numpy as np
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from transformers import AutoImageProcessor, Gemma3Processor

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.ltx2_text_encoder import LTX2TextEncoder, LTX2TextEncoderPostModules, LTXVGemmaTokenizer
from ..models.ltx2_dit import LTXModel
from ..models.ltx2_video_vae import LTX2VideoEncoder, LTX2VideoDecoder, VideoLatentPatchifier
from ..models.ltx2_audio_vae import LTX2AudioEncoder, LTX2AudioDecoder, LTX2Vocoder, AudioPatchifier, AudioProcessor
from ..models.ltx2_upsampler import LTX2LatentUpsampler
from ..models.ltx2_common import VideoLatentShape, AudioLatentShape, VideoPixelShape, get_pixel_coords, VIDEO_SCALE_FACTORS
from ..utils.data.media_io_ltx2 import ltx2_preprocess
from ..utils.data.audio import convert_to_stereo


class LTX2AudioVideoPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=32,
            width_division_factor=32,
            time_division_factor=8,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("LTX-2")
        self.text_encoder: LTX2TextEncoder = None
        self.tokenizer: LTXVGemmaTokenizer = None
        self.processor: Gemma3Processor = None
        self.text_encoder_post_modules: LTX2TextEncoderPostModules = None
        self.dit: LTXModel = None
        self.video_vae_encoder: LTX2VideoEncoder = None
        self.video_vae_decoder: LTX2VideoDecoder = None
        self.audio_vae_encoder: LTX2AudioEncoder = None
        self.audio_vae_decoder: LTX2AudioDecoder = None
        self.audio_vocoder: LTX2Vocoder = None
        self.upsampler: LTX2LatentUpsampler = None

        self.video_patchifier: VideoLatentPatchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier: AudioPatchifier = AudioPatchifier(patch_size=1)
        self.audio_processor: AudioProcessor = AudioProcessor()

        self.in_iteration_models = ("dit",)
        self.units = [
            LTX2AudioVideoUnit_PipelineChecker(),
            LTX2AudioVideoUnit_ShapeChecker(),
            LTX2AudioVideoUnit_PromptEmbedder(),
            LTX2AudioVideoUnit_NoiseInitializer(),
            LTX2AudioVideoUnit_VideoRetakeEmbedder(),
            LTX2AudioVideoUnit_AudioRetakeEmbedder(),
            LTX2AudioVideoUnit_InputAudioEmbedder(),
            LTX2AudioVideoUnit_InputVideoEmbedder(),
            LTX2AudioVideoUnit_InputImagesEmbedder(),
            LTX2AudioVideoUnit_InContextVideoEmbedder(),
        ]
        self.stage2_units = [
            LTX2AudioVideoUnit_SwitchStage2(),
            LTX2AudioVideoUnit_NoiseInitializer(),
            LTX2AudioVideoUnit_LatentsUpsampler(),
            LTX2AudioVideoUnit_VideoRetakeEmbedder(),
            LTX2AudioVideoUnit_AudioRetakeEmbedder(),
            LTX2AudioVideoUnit_InputImagesEmbedder(),
            LTX2AudioVideoUnit_SetScheduleStage2(),
        ]
        self.model_fn = model_fn_ltx2
        self.compilable_models = ["dit"]

        self.default_negative_prompt = {
            "LTX-2": (
                "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
                "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
                "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
                "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
                "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
                "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
                "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
                "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
                "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
                "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
                "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
            ),
            "LTX-2.3": (
                "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
                "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
                "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
                "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
                "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
                "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
                "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
                "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
                "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
                "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
                "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
            ),
        }

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
        stage2_lora_config: Optional[ModelConfig] = None,
        stage2_lora_strength: float = 0.8,
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = LTX2AudioVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("ltx2_text_encoder")
        tokenizer_config.download_if_necessary()
        pipe.tokenizer = LTXVGemmaTokenizer(tokenizer_path=tokenizer_config.path)
        image_processor = AutoImageProcessor.from_pretrained(tokenizer_config.path, local_files_only=True)
        pipe.processor = Gemma3Processor(image_processor=image_processor, tokenizer=pipe.tokenizer.tokenizer)

        pipe.text_encoder_post_modules = model_pool.fetch_model("ltx2_text_encoder_post_modules")
        pipe.dit = model_pool.fetch_model("ltx2_dit")
        pipe.video_vae_encoder = model_pool.fetch_model("ltx2_video_vae_encoder")
        pipe.video_vae_decoder = model_pool.fetch_model("ltx2_video_vae_decoder")
        pipe.audio_vae_decoder = model_pool.fetch_model("ltx2_audio_vae_decoder")
        pipe.audio_vocoder = model_pool.fetch_model("ltx2_audio_vocoder")
        pipe.upsampler = model_pool.fetch_model("ltx2_latent_upsampler")
        pipe.audio_vae_encoder = model_pool.fetch_model("ltx2_audio_vae_encoder")

        # Stage 2
        if stage2_lora_config is not None:
            pipe.stage2_lora_config = stage2_lora_config
            pipe.stage2_lora_strength = stage2_lora_strength

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    def denoise_stage(self, inputs_shared, inputs_posi, inputs_nega, units, cfg_scale=1.0, progress_bar_cmd=tqdm, skip_stage=False):
        if skip_stage:
            return inputs_shared, inputs_posi, inputs_nega
        for unit in units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_video, noise_pred_audio = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["video_latents"] = self.step(self.scheduler, inputs_shared["video_latents"], progress_id=progress_id, noise_pred=noise_pred_video,
                                                       inpaint_mask=inputs_shared.get("denoise_mask_video", None), input_latents=inputs_shared.get("input_latents_video", None), **inputs_shared)
            inputs_shared["audio_latents"] = self.step(self.scheduler, inputs_shared["audio_latents"], progress_id=progress_id, noise_pred=noise_pred_audio,
                                                       inpaint_mask=inputs_shared.get("denoise_mask_audio", None), input_latents=inputs_shared.get("input_latents_audio", None), **inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        denoising_strength: float = 1.0,
        # Image-to-video
        input_images: Optional[list[Image.Image]] = None,
        input_images_indexes: Optional[list[int]] = [0],
        input_images_strength: Optional[float] = 1.0,
        # In-Context Video Control
        in_context_videos: Optional[list[list[Image.Image]]] = None,
        in_context_downsample_factor: Optional[int] = 2,
        # Video-to-video
        retake_video: Optional[list[Image.Image]] = None,
        retake_video_regions: Optional[list[tuple[float, float]]] = None,
        # Audio-to-video
        retake_audio: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 48000,
        retake_audio_regions: Optional[list[tuple[float, float]]] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 512,
        width: Optional[int] = 768,
        num_frames: Optional[int] = 121,
        frame_rate: Optional[int] = 24,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 3.0,
        # Scheduler
        num_inference_steps: Optional[int] = 30,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size_in_pixels: Optional[int] = 512,
        tile_overlap_in_pixels: Optional[int] = 128,
        tile_size_in_frames: Optional[int] = 128,
        tile_overlap_in_frames: Optional[int] = 24,
        # Special Pipelines
        use_two_stage_pipeline: Optional[bool] = False,
        stage2_spatial_upsample_factor: Optional[int] = 2,
        clear_lora_before_state_two: Optional[bool] = False,
        use_distilled_pipeline: Optional[bool] = False,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, special_case="ditilled_stage1" if use_distilled_pipeline else None)
        # Inputs
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "input_images": input_images, "input_images_indexes": input_images_indexes, "input_images_strength": input_images_strength,
            "retake_video": retake_video, "retake_video_regions": retake_video_regions,
            "retake_audio": (retake_audio, audio_sample_rate) if retake_audio is not None else None, "retake_audio_regions": retake_audio_regions,
            "in_context_videos": in_context_videos, "in_context_downsample_factor": in_context_downsample_factor,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames, "frame_rate": frame_rate,
            "cfg_scale": cfg_scale,
            "tiled": tiled, "tile_size_in_pixels": tile_size_in_pixels, "tile_overlap_in_pixels": tile_overlap_in_pixels,
            "tile_size_in_frames": tile_size_in_frames, "tile_overlap_in_frames": tile_overlap_in_frames,
            "use_two_stage_pipeline": use_two_stage_pipeline, "use_distilled_pipeline": use_distilled_pipeline, "clear_lora_before_state_two": clear_lora_before_state_two, "stage2_spatial_upsample_factor": stage2_spatial_upsample_factor,
            "video_patchifier": self.video_patchifier, "audio_patchifier": self.audio_patchifier,
        }
        # Stage 1
        inputs_shared, inputs_posi, inputs_nega = self.denoise_stage(inputs_shared, inputs_posi, inputs_nega, self.units, cfg_scale, progress_bar_cmd)
        # Stage 2
        inputs_shared, inputs_posi, inputs_nega = self.denoise_stage(inputs_shared, inputs_posi, inputs_nega, self.stage2_units, 1.0, progress_bar_cmd, not inputs_shared["use_two_stage_pipeline"])
        # Decode
        self.load_models_to_device(['video_vae_decoder'])
        video = self.video_vae_decoder.decode(inputs_shared["video_latents"], tiled, tile_size_in_pixels, tile_overlap_in_pixels, tile_size_in_frames, tile_overlap_in_frames)
        video = self.vae_output_to_video(video)
        self.load_models_to_device(['audio_vae_decoder', 'audio_vocoder'])
        decoded_audio = self.audio_vae_decoder(inputs_shared["audio_latents"])
        decoded_audio = self.audio_vocoder(decoded_audio)
        decoded_audio = self.output_audio_format_check(decoded_audio)
        return video, decoded_audio


class LTX2AudioVideoUnit_PipelineChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("use_distilled_pipeline", "use_two_stage_pipeline"),
            output_params=("use_two_stage_pipeline", "cfg_scale")
        )

    def process(self, pipe: LTX2AudioVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("use_distilled_pipeline", False):
            inputs_shared["use_two_stage_pipeline"] = True
            inputs_shared["cfg_scale"] = 1.0
            print(f"Distilled pipeline requested, setting use_two_stage_pipeline to True, disable CFG by setting cfg_scale to 1.0.")
        if inputs_shared.get("use_two_stage_pipeline", False):
            # distill pipeline also uses two-stage, but it does not needs lora
            if not inputs_shared.get("use_distilled_pipeline", False):
                if not (hasattr(pipe, "stage2_lora_config") and pipe.stage2_lora_config is not None):
                    raise ValueError("Two-stage pipeline requested, but stage2_lora_config is not set in the pipeline.")
            if not (hasattr(pipe, "upsampler") and pipe.upsampler is not None):
                raise ValueError("Two-stage pipeline requested, but upsampler model is not loaded in the pipeline.")
        return inputs_shared, inputs_posi, inputs_nega


class LTX2AudioVideoUnit_ShapeChecker(PipelineUnit):
    """
    For two-stage pipelines, the resolution must be divisible by 64.
    For one-stage pipelines, the resolution must be divisible by 32.
    This unit set height and width to stage 1 resolution, and stage_2_width and stage_2_height.
    """
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "use_two_stage_pipeline", "stage2_spatial_upsample_factor"),
            output_params=("height", "width", "num_frames", "stage_2_height", "stage_2_width"),
        )

    def process(self, pipe: LTX2AudioVideoPipeline, height, width, num_frames, use_two_stage_pipeline=False, stage2_spatial_upsample_factor=2):
        if use_two_stage_pipeline:
            height, width = height // stage2_spatial_upsample_factor, width // stage2_spatial_upsample_factor
            height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
            stage_2_height, stage_2_width = int(height * stage2_spatial_upsample_factor), int(width * stage2_spatial_upsample_factor)
        else:
            stage_2_height, stage_2_width = None, None
            height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames, "stage_2_height": stage_2_height, "stage_2_width": stage_2_width}


class LTX2AudioVideoUnit_PromptEmbedder(PipelineUnit):

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("video_context", "audio_context"),
            onload_model_names=("text_encoder", "text_encoder_post_modules"),
        )
    def _preprocess_text(
        self,
        pipe,
        text: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        token_pairs = pipe.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=pipe.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=pipe.device)
        outputs = pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.hidden_states, attention_mask
    def encode_prompt(self, pipe, text, padding_side="left"):
        hidden_states, attention_mask = self._preprocess_text(pipe, text)
        video_encoding, audio_encoding, attention_mask = pipe.text_encoder_post_modules.process_hidden_states(
            hidden_states, attention_mask, padding_side)
        return video_encoding, audio_encoding, attention_mask

    def process(self, pipe: LTX2AudioVideoPipeline, prompt: str):
        pipe.load_models_to_device(self.onload_model_names)
        video_context, audio_context, _ = self.encode_prompt(pipe, prompt)
        return {"video_context": video_context, "audio_context": audio_context}


class LTX2AudioVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "frame_rate"),
            output_params=("video_noise", "audio_noise", "video_positions", "audio_positions", "video_latent_shape", "audio_latent_shape")
        )

    def process_stage(self, pipe: LTX2AudioVideoPipeline, height, width, num_frames, seed, rand_device, frame_rate=24.0):
        video_pixel_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        video_latent_shape = VideoLatentShape.from_pixel_shape(shape=video_pixel_shape, latent_channels=128)
        video_noise = pipe.generate_noise(video_latent_shape.to_torch_shape(), seed=seed, rand_device=rand_device)

        latent_coords = pipe.video_patchifier.get_patch_grid_bounds(output_shape=video_latent_shape, device=pipe.device)
        video_positions = get_pixel_coords(latent_coords, VIDEO_SCALE_FACTORS, True).float()
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / frame_rate
        video_positions = video_positions.to(pipe.torch_dtype)

        audio_latent_shape = AudioLatentShape.from_video_pixel_shape(video_pixel_shape)
        audio_noise = pipe.generate_noise(audio_latent_shape.to_torch_shape(), seed=seed, rand_device=rand_device)
        audio_positions = pipe.audio_patchifier.get_patch_grid_bounds(audio_latent_shape, device=pipe.device)
        return {
            "video_noise": video_noise,
            "audio_noise": audio_noise,
            "video_positions": video_positions,
            "audio_positions": audio_positions,
            "video_latent_shape": video_latent_shape,
            "audio_latent_shape": audio_latent_shape
        }

    def process(self, pipe: LTX2AudioVideoPipeline, height, width, num_frames, seed, rand_device, frame_rate=24.0):
        return self.process_stage(pipe, height, width, num_frames, seed, rand_device, frame_rate)


class LTX2AudioVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "video_noise", "tiled", "tile_size_in_pixels", "tile_overlap_in_pixels"),
            output_params=("video_latents", "input_latents"),
            onload_model_names=("video_vae_encoder")
        )

    def process(self, pipe: LTX2AudioVideoPipeline, input_video, video_noise, tiled, tile_size_in_pixels, tile_overlap_in_pixels):
        if input_video is None or not pipe.scheduler.training:
            return {"video_latents": video_noise}
        else:
            pipe.load_models_to_device(self.onload_model_names)
            input_video = pipe.preprocess_video(input_video)
            input_latents = pipe.video_vae_encoder.encode(input_video, tiled, tile_size_in_pixels, tile_overlap_in_pixels).to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"video_latents": input_latents, "input_latents": input_latents}

class LTX2AudioVideoUnit_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_audio", "audio_noise"),
            output_params=("audio_latents", "audio_input_latents", "audio_positions", "audio_latent_shape"),
            onload_model_names=("audio_vae_encoder",)
        )

    def process(self, pipe: LTX2AudioVideoPipeline, input_audio, audio_noise):
        if input_audio is None or not pipe.scheduler.training:
            return {"audio_latents": audio_noise}
        else:
            input_audio, sample_rate = input_audio
            input_audio = convert_to_stereo(input_audio)
            pipe.load_models_to_device(self.onload_model_names)
            input_audio = pipe.audio_processor.waveform_to_mel(input_audio.unsqueeze(0), waveform_sample_rate=sample_rate).to(dtype=pipe.torch_dtype)
            audio_input_latents = pipe.audio_vae_encoder(input_audio)
            audio_latent_shape = AudioLatentShape.from_torch_shape(audio_input_latents.shape)
            audio_positions = pipe.audio_patchifier.get_patch_grid_bounds(audio_latent_shape, device=pipe.device)
            return {"audio_latents": audio_input_latents, "audio_input_latents": audio_input_latents, "audio_positions": audio_positions, "audio_latent_shape": audio_latent_shape}


class LTX2AudioVideoUnit_VideoRetakeEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("retake_video", "height", "width", "tiled", "tile_size_in_pixels", "tile_overlap_in_pixels", "video_positions", "retake_video_regions"),
            output_params=("input_latents_video", "denoise_mask_video"),
            onload_model_names=("video_vae_encoder")
        )

    def process(self, pipe: LTX2AudioVideoPipeline, retake_video, height, width, tiled, tile_size_in_pixels, tile_overlap_in_pixels, video_positions, retake_video_regions=None):
        if retake_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        resized_video = [frame.resize((width, height)) for frame in retake_video]
        input_video = pipe.preprocess_video(resized_video)
        input_latents_video = pipe.video_vae_encoder.encode(input_video, tiled, tile_size_in_pixels, tile_overlap_in_pixels).to(dtype=pipe.torch_dtype, device=pipe.device)

        b, c, f, h, w = input_latents_video.shape
        denoise_mask_video = torch.zeros((b, 1, f, h, w), device=input_latents_video.device, dtype=input_latents_video.dtype)
        if retake_video_regions is not None and len(retake_video_regions) > 0:
            for start_time, end_time in retake_video_regions:
                t_start, t_end = video_positions[0, 0].unbind(dim=-1)
                in_region = (t_end >= start_time) & (t_start <= end_time)
                in_region = pipe.video_patchifier.unpatchify_video(in_region.unsqueeze(0).unsqueeze(-1), f, h, w)
                denoise_mask_video = torch.where(in_region, torch.ones_like(denoise_mask_video), denoise_mask_video)

        return {"input_latents_video": input_latents_video, "denoise_mask_video": denoise_mask_video}


class LTX2AudioVideoUnit_AudioRetakeEmbedder(PipelineUnit):
    """
    Functionality of audio2video, audio retaking.
    """
    def __init__(self):
        super().__init__(
            input_params=("retake_audio", "seed", "rand_device", "retake_audio_regions"),
            output_params=("input_latents_audio", "audio_noise", "audio_positions", "audio_latent_shape", "denoise_mask_audio"),
            onload_model_names=("audio_vae_encoder",)
        )

    def process(self, pipe: LTX2AudioVideoPipeline, retake_audio, seed, rand_device, retake_audio_regions=None):
        if retake_audio is None:
            return {}
        else:
            input_audio, sample_rate = retake_audio
            input_audio = convert_to_stereo(input_audio)
            pipe.load_models_to_device(self.onload_model_names)
            input_audio = pipe.audio_processor.waveform_to_mel(input_audio.unsqueeze(0), waveform_sample_rate=sample_rate).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents_audio = pipe.audio_vae_encoder(input_audio)
            audio_latent_shape = AudioLatentShape.from_torch_shape(input_latents_audio.shape)
            audio_positions = pipe.audio_patchifier.get_patch_grid_bounds(audio_latent_shape, device=pipe.device)
            # Regenerate noise for the new shape if retake_audio is provided, to avoid shape mismatch.
            audio_noise = pipe.generate_noise(input_latents_audio.shape, seed=seed, rand_device=rand_device)

            b, c, t, f = input_latents_audio.shape
            denoise_mask_audio = torch.zeros((b, 1, t, 1), device=input_latents_audio.device, dtype=input_latents_audio.dtype)
            if retake_audio_regions is not None and len(retake_audio_regions) > 0:
                for start_time, end_time in retake_audio_regions:
                    t_start, t_end = audio_positions[:, 0, :, 0], audio_positions[:, 0, :, 1]
                    in_region = (t_end >= start_time) & (t_start <= end_time)
                    in_region = pipe.audio_patchifier.unpatchify_audio(in_region.unsqueeze(-1), 1, 1)
                    denoise_mask_audio = torch.where(in_region, torch.ones_like(denoise_mask_audio), denoise_mask_audio)

            return {
                "input_latents_audio": input_latents_audio,
                "denoise_mask_audio": denoise_mask_audio,
                "audio_noise": audio_noise,
                "audio_positions": audio_positions,
                "audio_latent_shape": audio_latent_shape,
            }


class LTX2AudioVideoUnit_InputImagesEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_images", "input_images_indexes", "input_images_strength", "video_latents", "height", "width", "frame_rate", "tiled", "tile_size_in_pixels", "tile_overlap_in_pixels", "input_latents_video", "denoise_mask_video"),
            output_params=("denoise_mask_video", "input_latents_video", "ref_frames_latents", "ref_frames_positions"),
            onload_model_names=("video_vae_encoder")
        )

    def get_image_latent(self, pipe, input_image, height, width, tiled, tile_size_in_pixels, tile_overlap_in_pixels):
        image = ltx2_preprocess(np.array(input_image.resize((width, height))))
        image = torch.Tensor(np.array(image, dtype=np.float32)).to(dtype=pipe.torch_dtype, device=pipe.device)
        image = image / 127.5 - 1.0
        image = repeat(image, f"H W C -> B C F H W", B=1, F=1)
        latents = pipe.video_vae_encoder.encode(image, tiled, tile_size_in_pixels, tile_overlap_in_pixels).to(pipe.device)
        return latents

    def apply_input_images_to_latents(self, latents, input_latents, input_indexes, input_strength=1.0, input_latents_video=None, denoise_mask_video=None):
        b, _, f, h, w = latents.shape
        denoise_mask = torch.ones((b, 1, f, h, w), dtype=latents.dtype, device=latents.device) if denoise_mask_video is None else denoise_mask_video
        input_latents_video = torch.zeros_like(latents) if input_latents_video is None else input_latents_video
        for idx, input_latent in zip(input_indexes, input_latents):
            idx = min(max(1 + (idx-1) // 8, 0), f - 1)
            input_latent = input_latent.to(dtype=latents.dtype, device=latents.device)
            input_latents_video[:, :, idx:idx + input_latent.shape[2], :, :] = input_latent
            denoise_mask[:, :, idx:idx + input_latent.shape[2], :, :] = 1.0 - input_strength
        return input_latents_video, denoise_mask

    def process(
        self,
        pipe: LTX2AudioVideoPipeline,
        video_latents,
        input_images,
        height,
        width,
        frame_rate,
        tiled,
        tile_size_in_pixels,
        tile_overlap_in_pixels,
        input_images_indexes=[0],
        input_images_strength=1.0,
        input_latents_video=None,
        denoise_mask_video=None,
    ):
        if input_images is None or len(input_images) == 0:
            return {}
        else:
            if len(input_images_indexes) != len(set(input_images_indexes)):
                raise ValueError("Input images must have unique indexes.")
            pipe.load_models_to_device(self.onload_model_names)
            frame_conditions = {"input_latents_video": None, "denoise_mask_video": None, "ref_frames_latents": [], "ref_frames_positions": []}
            for img, index in zip(input_images, input_images_indexes):
                latents = self.get_image_latent(pipe, img, height, width, tiled, tile_size_in_pixels, tile_overlap_in_pixels)
                # first_frame by replacing latents
                if index == 0:
                    input_latents_video, denoise_mask_video = self.apply_input_images_to_latents(
                        video_latents, [latents], [0], input_images_strength, input_latents_video, denoise_mask_video)
                    frame_conditions.update({"input_latents_video": input_latents_video, "denoise_mask_video": denoise_mask_video})
                # other frames by adding reference latents
                else:
                    latent_coords = pipe.video_patchifier.get_patch_grid_bounds(output_shape=VideoLatentShape.from_torch_shape(latents.shape), device=pipe.device)
                    video_positions = get_pixel_coords(latent_coords, VIDEO_SCALE_FACTORS, False).float()
                    video_positions[:, 0, ...] = (video_positions[:, 0, ...] + index) / frame_rate
                    video_positions = video_positions.to(pipe.torch_dtype)
                    frame_conditions["ref_frames_latents"].append(latents)
                    frame_conditions["ref_frames_positions"].append(video_positions)
            if len(frame_conditions["ref_frames_latents"]) == 0:
                frame_conditions.update({"ref_frames_latents": None, "ref_frames_positions": None})
            return frame_conditions


class LTX2AudioVideoUnit_InContextVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("in_context_videos", "height", "width", "num_frames", "frame_rate", "in_context_downsample_factor", "tiled", "tile_size_in_pixels", "tile_overlap_in_pixels"),
            output_params=("in_context_video_latents", "in_context_video_positions"),
            onload_model_names=("video_vae_encoder")
        )

    def check_in_context_video(self, pipe, in_context_video, height, width, num_frames, in_context_downsample_factor):
        if in_context_video is None or len(in_context_video) == 0:
            raise ValueError("In-context video is None or empty.")
        in_context_video = in_context_video[:num_frames]
        expected_height = height // in_context_downsample_factor
        expected_width = width // in_context_downsample_factor
        current_h, current_w, current_f = in_context_video[0].size[1], in_context_video[0].size[0], len(in_context_video)
        h, w, f = pipe.check_resize_height_width(expected_height, expected_width, current_f, verbose=0)
        if current_h != h or current_w != w:
            in_context_video = [img.resize((w, h)) for img in in_context_video]
        if current_f != f:
            # pad black frames at the end
            in_context_video = in_context_video + [Image.new("RGB", (w, h), (0, 0, 0))] * (f - current_f)
        return in_context_video

    def process(self, pipe: LTX2AudioVideoPipeline, in_context_videos, height, width, num_frames, frame_rate, in_context_downsample_factor, tiled, tile_size_in_pixels, tile_overlap_in_pixels):
        if in_context_videos is None or len(in_context_videos) == 0:
            return {}
        else:
            pipe.load_models_to_device(self.onload_model_names)
            latents, positions = [], []
            for in_context_video in in_context_videos:
                in_context_video = self.check_in_context_video(pipe, in_context_video, height, width, num_frames, in_context_downsample_factor)
                in_context_video = pipe.preprocess_video(in_context_video)
                in_context_latents = pipe.video_vae_encoder.encode(in_context_video, tiled, tile_size_in_pixels, tile_overlap_in_pixels).to(dtype=pipe.torch_dtype, device=pipe.device)

                latent_coords = pipe.video_patchifier.get_patch_grid_bounds(output_shape=VideoLatentShape.from_torch_shape(in_context_latents.shape), device=pipe.device)
                video_positions = get_pixel_coords(latent_coords, VIDEO_SCALE_FACTORS, True).float()
                video_positions[:, 0, ...] = video_positions[:, 0, ...] / frame_rate
                video_positions[:, 1, ...] *= in_context_downsample_factor  # height axis
                video_positions[:, 2, ...] *= in_context_downsample_factor  # width axis
                video_positions = video_positions.to(pipe.torch_dtype)

                latents.append(in_context_latents)
                positions.append(video_positions)
            latents = torch.cat(latents, dim=1)
            positions = torch.cat(positions, dim=1)
            return {"in_context_video_latents": latents, "in_context_video_positions": positions}


class LTX2AudioVideoUnit_SwitchStage2(PipelineUnit):
    """
    1. switch height and width to stage 2 resolution
    2. clear in_context_video_latents and in_context_video_positions
    3. switch stage 2 lora model
    """
    def __init__(self):
        super().__init__(
            input_params=("stage_2_height", "stage_2_width", "clear_lora_before_state_two", "use_distilled_pipeline"),
            output_params=("height", "width", "in_context_video_latents", "in_context_video_positions"),
        )

    def process(self, pipe: LTX2AudioVideoPipeline, stage_2_height, stage_2_width, clear_lora_before_state_two, use_distilled_pipeline):
        stage2_params = {}
        stage2_params.update({"height": stage_2_height, "width": stage_2_width})
        stage2_params.update({"in_context_video_latents": None, "in_context_video_positions": None})
        stage2_params.update({"input_latents_video": None, "denoise_mask_video": None})
        if clear_lora_before_state_two:
            pipe.clear_lora()
        if not use_distilled_pipeline:
            pipe.load_lora(pipe.dit, pipe.stage2_lora_config, alpha=pipe.stage2_lora_strength, state_dict=pipe.stage2_lora_config.state_dict)
        return stage2_params


class LTX2AudioVideoUnit_SetScheduleStage2(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("video_latents", "video_noise", "audio_latents", "audio_noise"),
            output_params=("video_latents", "audio_latents"),
        )

    def process(self, pipe: LTX2AudioVideoPipeline, video_latents, video_noise, audio_latents, audio_noise):
        pipe.scheduler.set_timesteps(special_case="stage2")
        video_latents = pipe.scheduler.add_noise(video_latents, video_noise, pipe.scheduler.timesteps[0])
        audio_latents = pipe.scheduler.add_noise(audio_latents, audio_noise, pipe.scheduler.timesteps[0])
        return {"video_latents": video_latents, "audio_latents": audio_latents}


class LTX2AudioVideoUnit_LatentsUpsampler(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("video_latents",),
            output_params=("video_latents",),
            onload_model_names=("upsampler",),
        )

    def process(self, pipe: LTX2AudioVideoPipeline, video_latents):
        if video_latents is None or pipe.upsampler is None:
            raise ValueError("No upsampler or no video latents before stage 2.")
        else:
            pipe.load_models_to_device(self.onload_model_names)
            video_latents = pipe.video_vae_encoder.per_channel_statistics.un_normalize(video_latents)
            video_latents = pipe.upsampler(video_latents)
            video_latents = pipe.video_vae_encoder.per_channel_statistics.normalize(video_latents)
            return {"video_latents": video_latents}


def model_fn_ltx2(
    dit: LTXModel,
    video_latents=None,
    video_context=None,
    video_positions=None,
    video_patchifier=None,
    audio_latents=None,
    audio_context=None,
    audio_positions=None,
    audio_patchifier=None,
    timestep=None,
    # First Frame Conditioning
    input_latents_video=None,
    denoise_mask_video=None,
    # Other Frames Conditioning
    ref_frames_latents=None,
    ref_frames_positions=None,
    # In-Context Conditioning
    in_context_video_latents=None,
    in_context_video_positions=None,
    # Audio Inputs
    input_latents_audio=None,
    denoise_mask_audio=None,
    # Gradient Checkpointing
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    timestep = timestep.float() / 1000.

    # patchify
    b, c_v, f, h, w = video_latents.shape
    video_latents = video_patchifier.patchify(video_latents)
    seq_len_video = video_latents.shape[1]
    video_timesteps = timestep.repeat(1, video_latents.shape[1], 1)
    # Frist frame conditioning by replacing the video latents
    if input_latents_video is not None:
        denoise_mask_video = video_patchifier.patchify(denoise_mask_video)
        video_latents = video_latents * denoise_mask_video + video_patchifier.patchify(input_latents_video) * (1.0 - denoise_mask_video)
        video_timesteps = denoise_mask_video * video_timesteps

    # Reference conditioning by appending the reference video or frame latents
    total_ref_latents = ref_frames_latents if ref_frames_latents is not None else []
    total_ref_positions = ref_frames_positions if ref_frames_positions is not None else []
    total_ref_latents += [in_context_video_latents] if in_context_video_latents is not None else []
    total_ref_positions += [in_context_video_positions] if in_context_video_positions is not None else []
    if len(total_ref_latents) > 0:
        for ref_frames_latent, ref_frames_position in zip(total_ref_latents, total_ref_positions):
            ref_frames_latent = video_patchifier.patchify(ref_frames_latent)
            ref_frames_timestep = timestep.repeat(1, ref_frames_latent.shape[1], 1) * 0.
            video_latents = torch.cat([video_latents, ref_frames_latent], dim=1)
            video_positions = torch.cat([video_positions, ref_frames_position], dim=2)
            video_timesteps = torch.cat([video_timesteps, ref_frames_timestep], dim=1)

    if audio_latents is not None:
        _, c_a, _, mel_bins  = audio_latents.shape
        audio_latents = audio_patchifier.patchify(audio_latents)
        audio_timesteps = timestep.repeat(1, audio_latents.shape[1], 1)
    else:
        audio_timesteps = None
    if input_latents_audio is not None:
        denoise_mask_audio = audio_patchifier.patchify(denoise_mask_audio)
        audio_latents = audio_latents * denoise_mask_audio + audio_patchifier.patchify(input_latents_audio) * (1.0 - denoise_mask_audio)
        audio_timesteps = denoise_mask_audio * audio_timesteps

    vx, ax = dit(
        video_latents=video_latents,
        video_positions=video_positions,
        video_context=video_context,
        video_timesteps=video_timesteps,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_context=audio_context,
        audio_timesteps=audio_timesteps,
        sigma=timestep,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )

    vx = vx[:, :seq_len_video, ...]
    # unpatchify
    vx = video_patchifier.unpatchify_video(vx, f, h, w)
    ax = audio_patchifier.unpatchify_audio(ax, c_a, mel_bins) if ax is not None else None
    return vx, ax
