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
from ..models.ltx2_audio_vae import LTX2AudioEncoder, LTX2AudioDecoder, LTX2Vocoder, AudioPatchifier
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

        self.in_iteration_models = ("dit",)
        self.units = [
            LTX2AudioVideoUnit_PipelineChecker(),
            LTX2AudioVideoUnit_ShapeChecker(),
            LTX2AudioVideoUnit_PromptEmbedder(),
            LTX2AudioVideoUnit_NoiseInitializer(),
            LTX2AudioVideoUnit_InputVideoEmbedder(),
            LTX2AudioVideoUnit_InputImagesEmbedder(),
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

        # Stage 2
        if stage2_lora_config is not None:
            stage2_lora_config.download_if_necessary()
            pipe.stage2_lora_path = stage2_lora_config.path
        # Optional, currently not used
        # pipe.audio_vae_encoder = model_pool.fetch_model("ltx2_audio_vae_encoder")

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    def stage2_denoise(self, inputs_shared, inputs_posi, inputs_nega, progress_bar_cmd=tqdm):
        if inputs_shared["use_two_stage_pipeline"]:
            latent = self.video_vae_encoder.per_channel_statistics.un_normalize(inputs_shared["video_latents"])
            self.load_models_to_device('upsampler',)
            latent = self.upsampler(latent)
            latent = self.video_vae_encoder.per_channel_statistics.normalize(latent)
            self.scheduler.set_timesteps(special_case="stage2")
            inputs_shared.update({k.replace("stage2_", ""): v for k, v in inputs_shared.items() if k.startswith("stage2_")})
            denoise_mask_video = 1.0
            if inputs_shared.get("input_images", None) is not None:
                latent, denoise_mask_video, initial_latents = self.apply_input_images_to_latents(
                    latent, inputs_shared.pop("input_latents"), inputs_shared["input_images_indexes"],
                    inputs_shared["input_images_strength"], latent.clone())
                inputs_shared.update({"input_latents_video": initial_latents, "denoise_mask_video": denoise_mask_video})
            inputs_shared["video_latents"] = self.scheduler.sigmas[0] * denoise_mask_video * inputs_shared[
                "video_noise"] + (1 - self.scheduler.sigmas[0] * denoise_mask_video) * latent
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
        # Image-to-video
        denoising_strength: float = 1.0,
        input_images: Optional[list[Image.Image]] = None,
        input_images_indexes: Optional[list[int]] = None,
        input_images_strength: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 512,
        width: Optional[int] = 768,
        num_frames=121,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 3.0,
        cfg_merge: Optional[bool] = False,
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
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "tiled": tiled, "tile_size_in_pixels": tile_size_in_pixels, "tile_overlap_in_pixels": tile_overlap_in_pixels,
            "tile_size_in_frames": tile_size_in_frames, "tile_overlap_in_frames": tile_overlap_in_frames,
            "use_two_stage_pipeline": use_two_stage_pipeline, "use_distilled_pipeline": use_distilled_pipeline,
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

    def apply_input_images_to_latents(self, latents, input_latents, input_indexes, input_strength, initial_latents=None, num_frames=121):
        b, _, f, h, w = latents.shape
        denoise_mask = torch.ones((b, 1, f, h, w), dtype=latents.dtype, device=latents.device)
        initial_latents = torch.zeros_like(latents) if initial_latents is None else initial_latents
        for idx, input_latent in zip(input_indexes, input_latents):
            idx = min(max(1 + (idx-1) // 8, 0), f - 1)
            input_latent = input_latent.to(dtype=latents.dtype, device=latents.device)
            initial_latents[:, :, idx:idx + input_latent.shape[2], :, :] = input_latent
            denoise_mask[:, :, idx:idx + input_latent.shape[2], :, :] = 1.0 - input_strength
        latents = latents * denoise_mask + initial_latents * (1.0 - denoise_mask)
        return latents, denoise_mask, initial_latents


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

    def _convert_to_additive_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (attention_mask - 1).to(dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])) * torch.finfo(dtype).max

    def _run_connectors(self, pipe, encoded_input: torch.Tensor,
                        attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        connector_attention_mask = self._convert_to_additive_mask(attention_mask, encoded_input.dtype)

        encoded, encoded_connector_attention_mask = pipe.text_encoder_post_modules.embeddings_connector(
            encoded_input,
            connector_attention_mask,
        )

        # restore the mask values to int64
        attention_mask = (encoded_connector_attention_mask < 0.000001).to(torch.int64)
        attention_mask = attention_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
        encoded = encoded * attention_mask

        encoded_for_audio, _ = pipe.text_encoder_post_modules.audio_embeddings_connector(
            encoded_input, connector_attention_mask)

        return encoded, encoded_for_audio, attention_mask.squeeze(-1)

    def _norm_and_concat_padded_batch(
        self,
        encoded_text: torch.Tensor,
        sequence_lengths: torch.Tensor,
        padding_side: str = "right",
    ) -> torch.Tensor:
        """Normalize and flatten multi-layer hidden states, respecting padding.
        Performs per-batch, per-layer normalization using masked mean and range,
        then concatenates across the layer dimension.
        Args:
            encoded_text: Hidden states of shape [batch, seq_len, hidden_dim, num_layers].
            sequence_lengths: Number of valid (non-padded) tokens per batch item.
            padding_side: Whether padding is on "left" or "right".
        Returns:
            Normalized tensor of shape [batch, seq_len, hidden_dim * num_layers],
            with padded positions zeroed out.
        """
        b, t, d, l = encoded_text.shape  # noqa: E741
        device = encoded_text.device
        # Build mask: [B, T, 1, 1]
        token_indices = torch.arange(t, device=device)[None, :]  # [1, T]
        if padding_side == "right":
            # For right padding, valid tokens are from 0 to sequence_length-1
            mask = token_indices < sequence_lengths[:, None]  # [B, T]
        elif padding_side == "left":
            # For left padding, valid tokens are from (T - sequence_length) to T-1
            start_indices = t - sequence_lengths[:, None]  # [B, 1]
            mask = token_indices >= start_indices  # [B, T]
        else:
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
        mask = rearrange(mask, "b t -> b t 1 1")
        eps = 1e-6
        # Compute masked mean: [B, 1, 1, L]
        masked = encoded_text.masked_fill(~mask, 0.0)
        denom = (sequence_lengths * d).view(b, 1, 1, 1)
        mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)
        # Compute masked min/max: [B, 1, 1, L]
        x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
        x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
        range_ = x_max - x_min
        # Normalize only the valid tokens
        normed = 8 * (encoded_text - mean) / (range_ + eps)
        # concat to be [Batch, T,  D * L] - this preserves the original structure
        normed = normed.reshape(b, t, -1)  # [B, T, D * L]
        # Apply mask to preserve original padding (set padded positions to 0)
        mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
        normed = normed.masked_fill(~mask_flattened, 0.0)

        return normed

    def _run_feature_extractor(self,
                               pipe,
                               hidden_states: torch.Tensor,
                               attention_mask: torch.Tensor,
                               padding_side: str = "right") -> torch.Tensor:
        encoded_text_features = torch.stack(hidden_states, dim=-1)
        encoded_text_features_dtype = encoded_text_features.dtype
        sequence_lengths = attention_mask.sum(dim=-1)
        normed_concated_encoded_text_features = self._norm_and_concat_padded_batch(encoded_text_features,
                                                                                   sequence_lengths,
                                                                                   padding_side=padding_side)

        return pipe.text_encoder_post_modules.feature_extractor_linear(
            normed_concated_encoded_text_features.to(encoded_text_features_dtype))

    def _preprocess_text(
        self,
        pipe,
        text: str,
        padding_side: str = "left",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Encode a given string into feature tensors suitable for downstream tasks.
        Args:
            text (str): Input string to encode.
        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Encoded features and a dictionary with attention mask.
        """
        token_pairs = pipe.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=pipe.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=pipe.device)
        outputs = pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        projected = self._run_feature_extractor(pipe,
                                                hidden_states=outputs.hidden_states,
                                                attention_mask=attention_mask,
                                                padding_side=padding_side)
        return projected, attention_mask

    def encode_prompt(self, pipe, text, padding_side="left"):
        encoded_inputs, attention_mask = self._preprocess_text(pipe, text, padding_side)
        video_encoding, audio_encoding, attention_mask = self._run_connectors(pipe, encoded_inputs, attention_mask)
        return video_encoding, audio_encoding, attention_mask

    def process(self, pipe: LTX2AudioVideoPipeline, prompt: str):
        pipe.load_models_to_device(self.onload_model_names)
        video_context, audio_context, _ = self.encode_prompt(pipe, prompt)
        return {"video_context": video_context, "audio_context": audio_context}


class LTX2AudioVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "use_two_stage_pipeline"),
            output_params=("video_noise", "audio_noise",),
        )

    def process_stage(self, pipe: LTX2AudioVideoPipeline, height, width, num_frames, seed, rand_device, frame_rate=24.0):
        video_pixel_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        video_latent_shape = VideoLatentShape.from_pixel_shape(shape=video_pixel_shape, latent_channels=pipe.video_vae_encoder.latent_channels)
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
            input_params=("input_video", "video_noise", "audio_noise", "tiled", "tile_size", "tile_stride"),
            output_params=("video_latents", "audio_latents"),
            onload_model_names=("video_vae_encoder")
        )

    def process(self, pipe: LTX2AudioVideoPipeline, input_video, video_noise, audio_noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"video_latents": video_noise, "audio_latents": audio_noise}
        else:
            # TODO: implement video-to-video
            raise NotImplementedError("Video-to-video not implemented yet.")

class LTX2AudioVideoUnit_InputImagesEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_images", "input_images_indexes", "input_images_strength", "video_latents", "height", "width", "num_frames", "tiled", "tile_size_in_pixels", "tile_overlap_in_pixels", "use_two_stage_pipeline"),
            output_params=("video_latents"),
            onload_model_names=("video_vae_encoder")
        )

    def get_image_latent(self, pipe, input_image, height, width, tiled, tile_size_in_pixels, tile_overlap_in_pixels):
        image = ltx2_preprocess(np.array(input_image.resize((width, height))))
        image = torch.Tensor(np.array(image, dtype=np.float32)).to(dtype=pipe.torch_dtype, device=pipe.device)
        image = image / 127.5 - 1.0
        image = repeat(image, f"H W C -> B C F H W", B=1, F=1)
        latent = pipe.video_vae_encoder.encode(image, tiled, tile_size_in_pixels, tile_overlap_in_pixels).to(pipe.device)
        return latent

    def process(self, pipe: LTX2AudioVideoPipeline, input_images, input_images_indexes, input_images_strength, video_latents, height, width, num_frames, tiled, tile_size_in_pixels, tile_overlap_in_pixels, use_two_stage_pipeline=False):
        if input_images is None or len(input_images) == 0:
            return {"video_latents": video_latents}
        else:
            pipe.load_models_to_device(self.onload_model_names)
            output_dicts = {}
            stage1_height = height // 2 if use_two_stage_pipeline else height
            stage1_width = width // 2 if use_two_stage_pipeline else width
            stage1_latents = [
                self.get_image_latent(pipe, img, stage1_height, stage1_width, tiled, tile_size_in_pixels,
                                      tile_overlap_in_pixels) for img in input_images
            ]
            video_latents, denoise_mask_video, initial_latents = pipe.apply_input_images_to_latents(video_latents, stage1_latents, input_images_indexes, input_images_strength, num_frames=num_frames)
            output_dicts.update({"video_latents": video_latents, "denoise_mask_video": denoise_mask_video, "input_latents_video": initial_latents})
            if use_two_stage_pipeline:
                stage2_latents = [
                    self.get_image_latent(pipe, img, height, width, tiled, tile_size_in_pixels,
                                          tile_overlap_in_pixels) for img in input_images
                ]
                output_dicts.update({"stage2_input_latents": stage2_latents})
            return output_dicts


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
    denoise_mask_video=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    timestep = timestep.float() / 1000.

    # patchify
    b, c_v, f, h, w = video_latents.shape
    video_latents = video_patchifier.patchify(video_latents)
    video_timesteps = timestep.repeat(1, video_latents.shape[1], 1)
    if denoise_mask_video is not None:
        video_timesteps = video_patchifier.patchify(denoise_mask_video) * video_timesteps
    _, c_a, _, mel_bins  = audio_latents.shape
    audio_latents = audio_patchifier.patchify(audio_latents)
    audio_timesteps = timestep.repeat(1, audio_latents.shape[1], 1)
    #TODO: support gradient checkpointing in training
    vx, ax = dit(
        video_latents=video_latents,
        video_positions=video_positions,
        video_context=video_context,
        video_timesteps=video_timesteps,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_context=audio_context,
        audio_timesteps=audio_timesteps,
    )
    # unpatchify
    vx = video_patchifier.unpatchify_video(vx, f, h, w)
    ax = audio_patchifier.unpatchify_audio(ax, c_a, mel_bins)
    return vx, ax
