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
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.ltx2_text_encoder import LTX2TextEncoder, LTX2TextEncoderPostModules, LTXVGemmaTokenizer
from ..models.ltx2_dit import LTXModel
from ..models.ltx2_video_vae import LTX2VideoEncoder, LTX2VideoDecoder, VideoLatentPatchifier
from ..models.ltx2_audio_vae import LTX2AudioEncoder, LTX2AudioDecoder, LTX2Vocoder, AudioPatchifier, AudioProcessor
from ..models.ltx2_upsampler import LTX2LatentUpsampler
from ..models.ltx2_common import VideoLatentShape, AudioLatentShape, VideoPixelShape, get_pixel_coords, VIDEO_SCALE_FACTORS
from ..utils.data.media_io_ltx2 import ltx2_preprocess


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
            LTX2AudioVideoUnit_InputAudioEmbedder(),
            LTX2AudioVideoUnit_InputVideoEmbedder(),
            LTX2AudioVideoUnit_InputImagesEmbedder(),
            LTX2AudioVideoUnit_InContextVideoEmbedder(),
        ]
        self.model_fn = model_fn_ltx2

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
        stage2_lora_config: Optional[ModelConfig] = None,
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
            stage2_lora_config.download_if_necessary()
            pipe.stage2_lora_path = stage2_lora_config.path
        # Optional, currently not used

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    def stage2_denoise(self, inputs_shared, inputs_posi, inputs_nega, progress_bar_cmd=tqdm):
        if inputs_shared["use_two_stage_pipeline"]:
            if inputs_shared.get("clear_lora_before_state_two", False):
                self.clear_lora()
            latents = self.video_vae_encoder.per_channel_statistics.un_normalize(inputs_shared["video_latents"])
            self.load_models_to_device('upsampler',)
            latents = self.upsampler(latents)
            latents = self.video_vae_encoder.per_channel_statistics.normalize(latents)
            self.scheduler.set_timesteps(special_case="stage2")
            inputs_shared.update({k.replace("stage2_", ""): v for k, v in inputs_shared.items() if k.startswith("stage2_")})
            denoise_mask_video = 1.0
            # input image
            if inputs_shared.get("input_images", None) is not None:
                initial_latents, denoise_mask_video = self.apply_input_images_to_latents(latents, initial_latents=latents, **inputs_shared.get("stage2_input_latents_apply_kwargs", {}))
                inputs_shared.update({"input_latents_video": initial_latents, "denoise_mask_video": denoise_mask_video})
            # remove in-context video control in stage 2
            inputs_shared.pop("in_context_video_latents", None)
            inputs_shared.pop("in_context_video_positions", None)

            # initialize latents for stage 2
            inputs_shared["video_latents"] = self.scheduler.sigmas[0] * denoise_mask_video * inputs_shared[
                "video_noise"] + (1 - self.scheduler.sigmas[0] * denoise_mask_video) * latents
            inputs_shared["audio_latents"] = self.scheduler.sigmas[0] * inputs_shared["audio_noise"] + (
                1 - self.scheduler.sigmas[0]) * inputs_shared["audio_latents"]

            self.load_models_to_device(self.in_iteration_models)
            if not inputs_shared["use_distilled_pipeline"]:
                self.load_lora(self.dit, self.stage2_lora_path, alpha=0.8)
            models = {name: getattr(self, name) for name in self.in_iteration_models}
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                noise_pred_video, noise_pred_audio = self.cfg_guided_model_fn(
                    self.model_fn, 1.0, inputs_shared, inputs_posi, inputs_nega,
                    **models, timestep=timestep, progress_id=progress_id
                )
                inputs_shared["video_latents"] = self.step(self.scheduler, inputs_shared["video_latents"], progress_id=progress_id,
                                                           noise_pred=noise_pred_video, inpaint_mask=inputs_shared.get("denoise_mask_video", None),
                                                           input_latents=inputs_shared.get("input_latents_video", None), **inputs_shared)
                inputs_shared["audio_latents"] = self.step(self.scheduler, inputs_shared["audio_latents"], progress_id=progress_id,
                                                           noise_pred=noise_pred_audio, **inputs_shared)
        return inputs_shared

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
        num_inference_steps: Optional[int] = 40,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size_in_pixels: Optional[int] = 512,
        tile_overlap_in_pixels: Optional[int] = 128,
        tile_size_in_frames: Optional[int] = 128,
        tile_overlap_in_frames: Optional[int] = 24,
        # Special Pipelines
        use_two_stage_pipeline: Optional[bool] = False,
        clear_lora_before_state_two: Optional[bool] = False,
        use_distilled_pipeline: Optional[bool] = False,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength,
                                     special_case="ditilled_stage1" if use_distilled_pipeline else None)
        # Inputs
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "input_images": input_images, "input_images_indexes": input_images_indexes, "input_images_strength": input_images_strength,
            "in_context_videos": in_context_videos, "in_context_downsample_factor": in_context_downsample_factor,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames, "frame_rate": frame_rate,
            "cfg_scale": cfg_scale,
            "tiled": tiled, "tile_size_in_pixels": tile_size_in_pixels, "tile_overlap_in_pixels": tile_overlap_in_pixels,
            "tile_size_in_frames": tile_size_in_frames, "tile_overlap_in_frames": tile_overlap_in_frames,
            "use_two_stage_pipeline": use_two_stage_pipeline, "use_distilled_pipeline": use_distilled_pipeline, "clear_lora_before_state_two": clear_lora_before_state_two,
            "video_patchifier": self.video_patchifier, "audio_patchifier": self.audio_patchifier,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise Stage 1
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
            inputs_shared["audio_latents"] = self.step(self.scheduler, inputs_shared["audio_latents"], progress_id=progress_id,
                                                       noise_pred=noise_pred_audio, **inputs_shared)

        # Denoise Stage 2
        inputs_shared = self.stage2_denoise(inputs_shared, inputs_posi, inputs_nega, progress_bar_cmd)

        # Decode
        self.load_models_to_device(['video_vae_decoder'])
        video = self.video_vae_decoder.decode(inputs_shared["video_latents"], tiled, tile_size_in_pixels,
                                              tile_overlap_in_pixels, tile_size_in_frames, tile_overlap_in_frames)
        video = self.vae_output_to_video(video)
        self.load_models_to_device(['audio_vae_decoder', 'audio_vocoder'])
        decoded_audio = self.audio_vae_decoder(inputs_shared["audio_latents"])
        decoded_audio = self.audio_vocoder(decoded_audio).squeeze(0).float()
        return video, decoded_audio

    def apply_input_images_to_latents(self, latents, input_latents, input_indexes, input_strength=1.0, initial_latents=None, denoise_mask_video=None):
        b, _, f, h, w = latents.shape
        denoise_mask = torch.ones((b, 1, f, h, w), dtype=latents.dtype, device=latents.device) if denoise_mask_video is None else denoise_mask_video
        initial_latents = torch.zeros_like(latents) if initial_latents is None else initial_latents
        for idx, input_latent in zip(input_indexes, input_latents):
            idx = min(max(1 + (idx-1) // 8, 0), f - 1)
            input_latent = input_latent.to(dtype=latents.dtype, device=latents.device)
            initial_latents[:, :, idx:idx + input_latent.shape[2], :, :] = input_latent
            denoise_mask[:, :, idx:idx + input_latent.shape[2], :, :] = 1.0 - input_strength
        return initial_latents, denoise_mask


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
                if not (hasattr(pipe, "stage2_lora_path") and pipe.stage2_lora_path is not None):
                    raise ValueError("Two-stage pipeline requested, but stage2_lora_path is not set in the pipeline.")
            if not (hasattr(pipe, "upsampler") and pipe.upsampler is not None):
                raise ValueError("Two-stage pipeline requested, but upsampler model is not loaded in the pipeline.")
        return inputs_shared, inputs_posi, inputs_nega


class LTX2AudioVideoUnit_ShapeChecker(PipelineUnit):
    """
    For two-stage pipelines, the resolution must be divisible by 64.
    For one-stage pipelines, the resolution must be divisible by 32.
    """
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: LTX2AudioVideoPipeline, height, width, num_frames, use_two_stage_pipeline=False):
        if use_two_stage_pipeline:
            self.width_division_factor = 64
            self.height_division_factor = 64
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        if use_two_stage_pipeline:
            self.width_division_factor = 32
            self.height_division_factor = 32
        return {"height": height, "width": width, "num_frames": num_frames}


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
            input_params=("height", "width", "num_frames", "seed", "rand_device", "frame_rate", "use_two_stage_pipeline"),
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

    def process(self, pipe: LTX2AudioVideoPipeline, height, width, num_frames, seed, rand_device, frame_rate=24.0, use_two_stage_pipeline=False):
        if use_two_stage_pipeline:
            stage1_dict = self.process_stage(pipe, height // 2, width // 2, num_frames, seed, rand_device, frame_rate)
            stage2_dict = self.process_stage(pipe, height, width, num_frames, seed, rand_device, frame_rate)
            initial_dict = stage1_dict
            initial_dict.update({"stage2_" + k: v for k, v in stage2_dict.items()})
            return initial_dict
        else:
            return self.process_stage(pipe, height, width, num_frames, seed, rand_device, frame_rate)

class LTX2AudioVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "video_noise", "tiled", "tile_size_in_pixels", "tile_overlap_in_pixels"),
            output_params=("video_latents", "input_latents"),
            onload_model_names=("video_vae_encoder")
        )

    def process(self, pipe: LTX2AudioVideoPipeline, input_video, video_noise, tiled, tile_size_in_pixels, tile_overlap_in_pixels):
        if input_video is None:
            return {"video_latents": video_noise}
        else:
            pipe.load_models_to_device(self.onload_model_names)
            input_video = pipe.preprocess_video(input_video)
            input_latents = pipe.video_vae_encoder.encode(input_video, tiled, tile_size_in_pixels, tile_overlap_in_pixels).to(dtype=pipe.torch_dtype, device=pipe.device)
            if pipe.scheduler.training:
                return {"video_latents": input_latents, "input_latents": input_latents}
            else:
                raise NotImplementedError("Video-to-video not implemented yet.")

class LTX2AudioVideoUnit_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_audio", "audio_noise"),
            output_params=("audio_latents", "audio_input_latents", "audio_positions", "audio_latent_shape"),
            onload_model_names=("audio_vae_encoder",)
        )

    def process(self, pipe: LTX2AudioVideoPipeline, input_audio, audio_noise):
        if input_audio is None:
            return {"audio_latents": audio_noise}
        else:
            input_audio, sample_rate = input_audio
            pipe.load_models_to_device(self.onload_model_names)
            input_audio = pipe.audio_processor.waveform_to_mel(input_audio.unsqueeze(0), waveform_sample_rate=sample_rate).to(dtype=pipe.torch_dtype)
            audio_input_latents = pipe.audio_vae_encoder(input_audio)
            audio_latent_shape = AudioLatentShape.from_torch_shape(audio_input_latents.shape)
            audio_positions = pipe.audio_patchifier.get_patch_grid_bounds(audio_latent_shape, device=pipe.device)
            if pipe.scheduler.training:
                return {"audio_latents": audio_input_latents, "audio_input_latents": audio_input_latents, "audio_positions": audio_positions, "audio_latent_shape": audio_latent_shape}
            else:
                raise NotImplementedError("Audio-to-video not supported.")

class LTX2AudioVideoUnit_InputImagesEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_images", "input_images_indexes", "input_images_strength", "video_latents", "height", "width", "tiled", "tile_size_in_pixels", "tile_overlap_in_pixels", "use_two_stage_pipeline"),
            output_params=("denoise_mask_video", "input_latents_video", "stage2_input_latents_apply_kwargs"),
            onload_model_names=("video_vae_encoder")
        )

    def get_image_latent(self, pipe, input_image, height, width, tiled, tile_size_in_pixels, tile_overlap_in_pixels):
        image = ltx2_preprocess(np.array(input_image.resize((width, height))))
        image = torch.Tensor(np.array(image, dtype=np.float32)).to(dtype=pipe.torch_dtype, device=pipe.device)
        image = image / 127.5 - 1.0
        image = repeat(image, f"H W C -> B C F H W", B=1, F=1)
        latents = pipe.video_vae_encoder.encode(image, tiled, tile_size_in_pixels, tile_overlap_in_pixels).to(pipe.device)
        return latents

    def get_frame_conditions(self, pipe: LTX2AudioVideoPipeline, input_images, input_images_indexes, input_images_strength, height, width, tiled, tile_size_in_pixels, tile_overlap_in_pixels, video_latents=None, skip_apply=False):
        frame_conditions = {}
        for img, index in zip(input_images, input_images_indexes):
            latents = self.get_image_latent(pipe, img, height, width, tiled, tile_size_in_pixels, tile_overlap_in_pixels)
            # first_frame
            if index == 0:
                if skip_apply:
                    frame_conditions = {"input_latents": [latents], "input_indexes": [0], "input_strength": input_images_strength}
                else:
                    input_latents_video, denoise_mask_video = pipe.apply_input_images_to_latents(video_latents, [latents], [0], input_images_strength)
                    frame_conditions.update({"input_latents_video": input_latents_video, "denoise_mask_video": denoise_mask_video})
        return frame_conditions

    def process(self, pipe: LTX2AudioVideoPipeline, input_images, input_images_indexes, input_images_strength, video_latents, height, width, tiled, tile_size_in_pixels, tile_overlap_in_pixels, use_two_stage_pipeline=False):
        if input_images is None or len(input_images) == 0:
            return {}
        else:
            if len(input_images_indexes) != len(set(input_images_indexes)):
                raise ValueError("Input images must have unique indexes.")
            pipe.load_models_to_device(self.onload_model_names)
            output_dicts = {}
            stage1_height = height // 2 if use_two_stage_pipeline else height
            stage1_width = width // 2 if use_two_stage_pipeline else width
            stage_1_frame_conditions = self.get_frame_conditions(pipe, input_images, input_images_indexes, input_images_strength, stage1_height, stage1_width,
                                                                 tiled, tile_size_in_pixels, tile_overlap_in_pixels, video_latents)
            output_dicts.update(stage_1_frame_conditions)
            if use_two_stage_pipeline:
                stage2_input_latents_apply_kwargs = self.get_frame_conditions(pipe, input_images, input_images_indexes, input_images_strength, height, width, 
                                                                              tiled, tile_size_in_pixels, tile_overlap_in_pixels, skip_apply=True)
                output_dicts.update({"stage2_input_latents_apply_kwargs": stage2_input_latents_apply_kwargs})
            return output_dicts


class LTX2AudioVideoUnit_InContextVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("in_context_videos", "height", "width", "num_frames", "frame_rate", "in_context_downsample_factor", "tiled", "tile_size_in_pixels", "tile_overlap_in_pixels", "use_two_stage_pipeline"),
            output_params=("in_context_video_latents", "in_context_video_positions"),
            onload_model_names=("video_vae_encoder")
        )

    def check_in_context_video(self, pipe, in_context_video, height, width, num_frames, in_context_downsample_factor, use_two_stage_pipeline=True):
        if in_context_video is None or len(in_context_video) == 0:
            raise ValueError("In-context video is None or empty.")
        in_context_video = in_context_video[:num_frames]
        expected_height = height // in_context_downsample_factor // 2 if use_two_stage_pipeline else height // in_context_downsample_factor
        expected_width = width // in_context_downsample_factor // 2 if use_two_stage_pipeline else width // in_context_downsample_factor
        current_h, current_w, current_f = in_context_video[0].size[1], in_context_video[0].size[0], len(in_context_video)
        h, w, f = pipe.check_resize_height_width(expected_height, expected_width, current_f, verbose=0)
        if current_h != h or current_w != w:
            in_context_video = [img.resize((w, h)) for img in in_context_video]
        if current_f != f:
            # pad black frames at the end
            in_context_video = in_context_video + [Image.new("RGB", (w, h), (0, 0, 0))] * (f - current_f)
        return in_context_video

    def process(self, pipe: LTX2AudioVideoPipeline, in_context_videos, height, width, num_frames, frame_rate, in_context_downsample_factor, tiled, tile_size_in_pixels, tile_overlap_in_pixels, use_two_stage_pipeline=True):
        if in_context_videos is None or len(in_context_videos) == 0:
            return {}
        else:
            pipe.load_models_to_device(self.onload_model_names)
            latents, positions = [], []
            for in_context_video in in_context_videos:
                in_context_video = self.check_in_context_video(pipe, in_context_video, height, width, num_frames, in_context_downsample_factor, use_two_stage_pipeline)
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
    input_latents_video=None,
    denoise_mask_video=None,
    in_context_video_latents=None,
    in_context_video_positions=None,
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
    if denoise_mask_video is not None:
        denoise_mask_video = video_patchifier.patchify(denoise_mask_video)
        video_latents = video_latents * denoise_mask_video + video_patchifier.patchify(input_latents_video) * (1.0 - denoise_mask_video)
        video_timesteps = denoise_mask_video * video_timesteps

    if in_context_video_latents is not None:
        in_context_video_latents = video_patchifier.patchify(in_context_video_latents)
        in_context_video_timesteps = timestep.repeat(1, in_context_video_latents.shape[1], 1) * 0.
        video_latents = torch.cat([video_latents, in_context_video_latents], dim=1)
        video_positions = torch.cat([video_positions, in_context_video_positions], dim=2)
        video_timesteps = torch.cat([video_timesteps, in_context_video_timesteps], dim=1)

    if audio_latents is not None:
        _, c_a, _, mel_bins  = audio_latents.shape
        audio_latents = audio_patchifier.patchify(audio_latents)
        audio_timesteps = timestep.repeat(1, audio_latents.shape[1], 1)
    else:
        audio_timesteps = None

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
