import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.minimax_h3_dit import (
    MiniMaxH3DiT,
    patchify_video,
    unpatchify_video,
    pack_audio,
    unpack_audio,
)
from ..models.minimax_h3_text_encoder import MiniMaxH3TextEncoder
from ..models.minimax_h3_video_vae import MiniMaxH3VideoVAE, _VIDEO_LATENTS_MEAN, _VIDEO_LATENTS_STD
from ..models.minimax_h3_audio_vae import MiniMaxH3AudioVAE, _AUDIO_LATENTS_MEAN, _AUDIO_LATENTS_STD


class MiniMaxH3Pipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=32, width_division_factor=32,
        )
        self.scheduler = FlowMatchScheduler("MiniMax-H3")
        self.scheduler_audio = FlowMatchScheduler("MiniMax-H3")
        self.text_encoder: MiniMaxH3TextEncoder = None
        self.dit: MiniMaxH3DiT = None
        self.video_vae: MiniMaxH3VideoVAE = None
        self.audio_vae: MiniMaxH3AudioVAE = None
        self.tokenizer = None
        self.in_iteration_models = ("dit",)
        self.units = [
            MiniMaxH3Unit_ShapeChecker(),
            MiniMaxH3Unit_PromptEmbedder(),
            MiniMaxH3Unit_NoiseInitializer(),
            MiniMaxH3Unit_KeyframeEncoder(),
            MiniMaxH3Unit_ReferenceEncoder(),
            MiniMaxH3Unit_PackedSequenceBuilder(),
        ]
        self.model_fn = model_fn_minimax_h3
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = MiniMaxH3Pipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("minimax_h3_text_encoder")
        pipe.dit = model_pool.fetch_model("minimax_h3_dit")
        pipe.video_vae = model_pool.fetch_model("minimax_h3_video_vae")
        pipe.audio_vae = model_pool.fetch_model("minimax_h3_audio_vae")
        if pipe.audio_vae is not None and hasattr(pipe.audio_vae, "remove_weight_norm"):
            pipe.audio_vae.remove_weight_norm()
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import AutoTokenizer
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = 768,
        width: int = 1344,
        num_frames: int = 24,
        num_inference_steps: int = 50,
        seed: int = 42,
        rand_device: str = "cpu",
        cfg_scale: float = 1.0,
        flow_shift: float = 12.0,
        audio_flow_shift: float = 3.0,
        tiled: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 64,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        # FL2AV parameters (None = t2va mode)
        keyframes=None,
        keyframe_indices=None,
        imgvid_cond_noise_aug: float = 0.999,
        # Ref2AV parameters (None = non-ref2av mode)
        references=None,
        audio_cond_noise_aug: float = 1.0,
        progress_bar_cmd=tqdm,
    ):
        # 1. Schedulers (video / audio independent shift)
        self.scheduler.set_timesteps(num_inference_steps, shift=flow_shift)
        self.scheduler_audio.set_timesteps(num_inference_steps, shift=audio_flow_shift)

        # 2. Three-dict inputs. All feature params go into inputs_shared; Units
        # check None internally to decide whether to execute.
        inputs_posi = {}
        inputs_nega = {}
        inputs_shared = {
            "prompt": prompt,
            "height": height, "width": width, "num_frames": num_frames,
            "seed": seed, "rand_device": rand_device,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": use_gradient_checkpointing_offload,
            "keyframes": keyframes,
            "keyframe_indices": keyframe_indices,
            "imgvid_cond_noise_aug": imgvid_cond_noise_aug,
            "references": references,
            "audio_cond_noise_aug": audio_cond_noise_aug,
        }

        # 3. Unit chain (all units always run; units check None inputs to skip)
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # 4. Denoise loop
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, _ in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            t_video = float(1.0 - self.scheduler.sigmas[progress_id])
            t_audio = float(1.0 - self.scheduler_audio.sigmas[progress_id])
            noise_pred_video, noise_pred_audio = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega,
                **models, t_video=t_video, t_audio=t_audio,
                device=self.device, torch_dtype=self.torch_dtype,
            )
            inputs_shared["video_latents"] = self.step(
                self.scheduler, inputs_shared["video_latents"], progress_id, noise_pred=noise_pred_video,
            )
            inputs_shared["audio_latents"] = self.step(
                self.scheduler_audio, inputs_shared["audio_latents"], progress_id, noise_pred=noise_pred_audio,
            )

        # 5. Decode
        self.load_models_to_device(["video_vae"])
        frames = self.video_vae.decode_video(
            inputs_shared["video_latents"], tiled=tiled, tile_size=tile_size, tile_overlap=tile_overlap,
        )
        video = self.vae_output_to_video(frames, min_value=0, max_value=1)

        self.load_models_to_device(["audio_vae"])
        waveform = self.audio_vae.decode_audio(inputs_shared["audio_latents"])
        audio = self.output_audio_format_check(waveform)
        return video, audio


class MiniMaxH3Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "video_latent_t", "audio_latent_t", "latent_h", "latent_w"),
        )

    @staticmethod
    def _align_frame_count(frame_count: int) -> int:
        current = max(int(frame_count), 1)
        while current % 17 != 5:
            current += 1
        return current

    @staticmethod
    def _video_latent_t(frame_count: int) -> int:
        if frame_count <= 5:
            return 2
        return ((int(frame_count) - 5) // 17) * 5 + 2

    def process(self, pipe: MiniMaxH3Pipeline, height, width, num_frames):
        height, width = pipe.check_resize_height_width(height, width)
        frame_count = self._align_frame_count(num_frames)
        video_latent_t = self._video_latent_t(frame_count)
        audio_latent_t = int(round(float(num_frames) / 24.0 * 40.0))
        return {
            "height": height, "width": width,
            "video_latent_t": video_latent_t, "audio_latent_t": audio_latent_t,
            "latent_h": height // 16, "latent_w": width // 16,
        }


class MiniMaxH3Unit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("prompt",),
            output_params=("prompt_embeds",),
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe: MiniMaxH3Pipeline, prompt):
        pipe.load_models_to_device(("text_encoder",))
        input_ids = torch.tensor(
            pipe.tokenizer(text=prompt, add_special_tokens=False)["input_ids"], dtype=torch.long
        )[None].to(pipe.device)
        attn = torch.ones_like(input_ids)
        hidden = pipe.text_encoder(input_ids=input_ids, attention_mask=attn)
        prompt_embeds = hidden[0].to(pipe.device, torch.bfloat16)
        return {"prompt_embeds": prompt_embeds}


class MiniMaxH3Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("seed", "video_latent_t", "latent_h", "latent_w", "audio_latent_t", "rand_device"),
            output_params=("video_latents", "audio_latents"),
        )

    def process(self, pipe: MiniMaxH3Pipeline, seed, video_latent_t, latent_h, latent_w, audio_latent_t, rand_device):
        if seed is None:
            seed = 42
        video_latents = pipe.generate_noise(
            (1, 24, video_latent_t, latent_h, latent_w),
            seed=seed, rand_device=rand_device, rand_torch_dtype=torch.float32, torch_dtype=torch.float32,
        )
        audio_rows = pipe.generate_noise(
            (audio_latent_t * 2, 32),
            seed=seed, rand_device=rand_device, rand_torch_dtype=torch.float32, torch_dtype=torch.float32,
        )
        audio_latents = unpack_audio(audio_rows, audio_channel=2, steps=audio_latent_t)
        return {"video_latents": video_latents, "audio_latents": audio_latents}


_MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42
# Ref: target reference_encoding.py:45-48
_MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
_MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE = 32
_MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
_MINIMAX_H3_AUDIO_CHANNELS = 2
# Ref: target reference_encoding.py:437-438
_MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED = 42
_MINIMAX_H3_REFERENCE_VIDEO_PATCH_SIZE = (1, 2, 2)
# Ref: target constants.py:23
_MINIMAX_H3_SUPPORTED_FPS = 24


class _AudioVAEDeterminismContext:
    """Scoped determinism config for the audio encode.

    Ref: target reference_encoding.py:51-97 `_AudioVAEDeterminismContext`.
    Disables TF32, forces deterministic algorithms, DISABLES cuDNN entirely
    for the encode (convs run on the fallback kernels), and pins SDP to the
    math backend. Restored on exit so the decode path keeps its own config.
    """

    def __enter__(self):
        b = torch.backends
        self._saved = (
            b.cuda.matmul.allow_tf32,
            b.cudnn.allow_tf32,
            b.cudnn.benchmark,
            b.cudnn.deterministic,
            b.cudnn.enabled,
            b.cuda.flash_sdp_enabled(),
            b.cuda.mem_efficient_sdp_enabled(),
            b.cuda.math_sdp_enabled(),
        )
        b.cuda.matmul.allow_tf32 = False
        b.cudnn.allow_tf32 = False
        b.cudnn.benchmark = False
        b.cudnn.deterministic = True
        b.cudnn.enabled = False
        b.cuda.enable_flash_sdp(False)
        b.cuda.enable_mem_efficient_sdp(False)
        b.cuda.enable_math_sdp(True)
        return self

    def __exit__(self, exc_type, exc, tb):
        b = torch.backends
        (
            b.cuda.matmul.allow_tf32,
            b.cudnn.allow_tf32,
            b.cudnn.benchmark,
            b.cudnn.deterministic,
            b.cudnn.enabled,
            flash,
            mem_eff,
            math_sdp,
        ) = self._saved
        b.cuda.enable_flash_sdp(flash)
        b.cuda.enable_mem_efficient_sdp(mem_eff)
        b.cuda.enable_math_sdp(math_sdp)


class MiniMaxH3Unit_KeyframeEncoder(PipelineUnit):
    """Encode keyframe images for FL2AV conditioning. If keyframes is None, this
    unit is a no-op (t2va mode). Output: keyframe_cond_anchor (patchified rows
    with noise augmentation applied once, used as fixed anchor each denoise step)."""

    def __init__(self):
        super().__init__(
            input_params=("keyframes", "keyframe_indices", "latent_h", "latent_w",
                          "video_latent_t", "seed", "imgvid_cond_noise_aug"),
            output_params=("keyframe_cond_anchor", "keyframe_indices_validated"),
            onload_model_names=("video_vae",),
        )

    def process(self, pipe: MiniMaxH3Pipeline, keyframes, keyframe_indices, latent_h, latent_w,
                video_latent_t, seed, imgvid_cond_noise_aug):
        if keyframes is None:
            return {}
        if keyframe_indices is None:
            raise ValueError("keyframe_indices must be provided when keyframes is not None")
        if len(keyframes) != len(keyframe_indices):
            raise ValueError(f"len(keyframes)={len(keyframes)} != len(keyframe_indices)={len(keyframe_indices)}")
        for idx in keyframe_indices:
            if idx not in (0, -1):
                raise ValueError(f"keyframe_indices must be 0 or -1, got {idx}")

        pipe.load_models_to_device(("video_vae",))
        device = pipe.device

        # Encode each keyframe: PIL → video_vae.encode_images → latent [24,1,H',W']
        # Then normalize (z - mean) / std, patchify to rows [frame_rows, 96]
        mean = torch.tensor(_VIDEO_LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(_VIDEO_LATENTS_STD, dtype=torch.float32).view(1, -1, 1, 1, 1)

        all_cond_rows = []
        for img in keyframes:
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
                target_w, target_h = latent_w * 16, latent_h * 16
                img_resized = img.resize((target_w, target_h), Image.LANCZOS)
            else:
                img_resized = img

            # Encode with fixed seed=42 fork (target library convention).
            # We manually preprocess + call encode_base to control dtype (VRAM
            # management casts VAE weights to bf16, so input must match).
            img_np = np.array(img_resized)  # [H, W, 3] uint8
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1,3,H,W]
            img_tensor = pipe.video_vae.processor.transform_tensor(img_tensor)  # ImageNet normalize
            vae_dtype = next(pipe.video_vae.parameters()).dtype
            img_tensor = img_tensor.to(device=device, dtype=vae_dtype)
            with torch.random.fork_rng(devices=[device] if str(device) != "cpu" else []):
                torch.manual_seed(_MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                z = pipe.video_vae.encode_base(img_tensor, process_image=True)  # [1,24,1,H',W']
            # Normalize: (z - mean) / std
            z_norm = (z - mean.to(z.device)) / std.to(z.device)
            # Patchify [1,2,2] → [frame_rows, 96]
            rows = patchify_video(z_norm)
            all_cond_rows.append(rows)

        # Concatenate all keyframe rows to clean anchor (fp32, on device).
        clean_cond_rows = torch.cat(all_cond_rows, dim=0).to(device=device, dtype=torch.float32)

        # Noise augmentation matches target condition_noise.py:24-119
        # `minimax_h3_imgvid_cond_noise_aug_rows`. Semantics:
        #  - each keyframe is a "condition" with shape (latent_t=1, latent_h, latent_w)
        #  - imgvid_cond_num_frames = number of keyframes
        #  - per condition, a FRESH CPU generator is created with the SAME seed
        #    and a `randn` of length `target_latent_t + imgvid_cond_num_frames`
        #    is drawn, then sliced `[:, :, :latent_t]` (prefix)
        #  - noise_aug == 1.0 short-circuits to the clean rows (no noise)
        seed_val = int(seed) if seed is not None else 42
        num_cond_frames = len(keyframes)
        noise_aug = float(imgvid_cond_noise_aug)
        if noise_aug == 1.0:
            cond_anchor = clean_cond_rows
        else:
            frame_rows = (latent_h // 2) * (latent_w // 2)
            timestep_tensor = torch.tensor(noise_aug, dtype=torch.float32, device=device)
            parts = []
            for i in range(num_cond_frames):
                latent_t_i = 1  # image keyframe = single-frame latent
                full_t = int(video_latent_t) + num_cond_frames
                generator = torch.Generator(device="cpu").manual_seed(seed_val)
                noise = torch.randn(
                    1, 24, full_t, latent_h, latent_w,
                    generator=generator, dtype=torch.float32, device="cpu",
                )[:, :, :latent_t_i]
                noise_rows = patchify_video(noise).to(device=device, dtype=torch.float32)
                clean_part = clean_cond_rows[i * frame_rows: (i + 1) * frame_rows]
                parts.append(timestep_tensor * clean_part + (1.0 - timestep_tensor) * noise_rows)
            cond_anchor = torch.cat(parts, dim=0).contiguous()

        return {
            "keyframe_cond_anchor": cond_anchor,
            "keyframe_indices_validated": list(keyframe_indices),
        }


class MiniMaxH3Unit_ReferenceEncoder(PipelineUnit):
    """Encode reference blocks for Ref2AV conditioning. References is a list of
    dicts, each with `type` in {"image", "video", "audio", "video_audio"} and
    modality-specific data. If references is None, this unit is a no-op.

    Output: `ref_blocks` — a list of dicts, each carrying the (already
    noise-augmented) anchor rows plus geometry needed by PackedSequenceBuilder
    and model_fn."""

    def __init__(self):
        super().__init__(
            input_params=("references", "seed", "imgvid_cond_noise_aug", "audio_cond_noise_aug", "video_latent_t"),
            output_params=("ref_blocks",),
            onload_model_names=("video_vae", "audio_vae"),
        )

    @staticmethod
    def _round_to_multiple(x: int, m: int = 32) -> int:
        return max(m, (x // m) * m)

    @staticmethod
    def _nearest_multiple(value: float, multiple: int) -> int:
        """Ref: target reference_encoding.py:100-101 `_nearest_multiple`."""
        return max(multiple, int(round(float(value) / multiple)) * multiple)

    def _encode_image_ref(self, pipe, img: Image.Image, seed_val: int):
        """Encode a reference image per target library.

        Ref: target reference_encoding.py:104-145 `minimax_h3_resolve_reference_image_shape`
             + reference_encoding.py:148-183 `minimax_h3_resize_reference_image`
             + keyframe_encoding.py:35-75 `minimax_h3_encode_keyframe_cond_rows`
             (image and keyframe share the same encode recipe).

        Resolution rule: short_edge=2048 with nearest-32 rounding per axis,
        aspect ratio 1:4 to 4:1 guard, upscale allowed."""
        img = img.convert("RGB")
        src_w, src_h = img.size
        if float(src_w) > 4.0 * float(src_h) or float(src_h) > 4.0 * float(src_w):
            raise ValueError(
                f"reference image ratio must be within 1:4 to 4:1, got {src_w}x{src_h}"
            )
        scale = _MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(float(src_w), float(src_h))
        target_w = self._nearest_multiple(float(src_w) * scale, _MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE)
        target_h = self._nearest_multiple(float(src_h) * scale, _MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE)
        if (target_w, target_h) != (src_w, src_h):
            img = img.resize((target_w, target_h), Image.LANCZOS)

        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = pipe.video_vae.processor.transform_tensor(img_tensor)
        vae_dtype = next(pipe.video_vae.parameters()).dtype
        img_tensor = img_tensor.to(device=pipe.device, dtype=vae_dtype)
        with torch.random.fork_rng(devices=[pipe.device] if str(pipe.device) != "cpu" else []):
            torch.manual_seed(_MINIMAX_H3_KEYFRAME_ENCODE_SEED)
            z = pipe.video_vae.encode_base(img_tensor, process_image=True)  # [1,24,1,H',W']
        mean = torch.tensor(_VIDEO_LATENTS_MEAN, device=z.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(_VIDEO_LATENTS_STD, device=z.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        z_norm = (z.float() - mean) / std
        rows = patchify_video(z_norm)
        return rows, int(z.shape[-2]), int(z.shape[-1])

    def _encode_video_ref(self, pipe, video_frames, seed_val: int, target_frame_count: int = None):
        """Encode a reference video per target library.

        Ref: target reference_encoding.py:379-499 `minimax_h3_prepare_reference_video`
             + `minimax_h3_encode_reference_video_rows`.

        Frame count rule: `target_frame_count` (17n+5 from target duration).
        Truncate leading frames only when source > target (target-lib docstring).
        Otherwise keep source frames as-is.

        Spatial rule: target library uses admission-time resolved shape (pre-queue
        pass). We approximate by nearest-32 rounding of source dims.
        """
        if isinstance(video_frames, (list, tuple)):
            if isinstance(video_frames[0], Image.Image):
                video_frames = [np.array(f.convert("RGB")) for f in video_frames]
            frames_np = np.stack(video_frames, axis=0)
        elif isinstance(video_frames, torch.Tensor):
            frames_np = video_frames.cpu().numpy()
        else:
            frames_np = np.asarray(video_frames)

        # Truncate to target_frame_count only if source is longer.
        if target_frame_count is not None and frames_np.shape[0] > int(target_frame_count):
            frames_np = frames_np[: int(target_frame_count)]

        # Spatial resize: nearest-32 per axis (approx for admission-time resolver).
        h_orig, w_orig = int(frames_np.shape[1]), int(frames_np.shape[2])
        new_h = self._nearest_multiple(h_orig, _MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE)
        new_w = self._nearest_multiple(w_orig, _MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE)
        if (new_h, new_w) != (h_orig, w_orig):
            resized = []
            for i in range(frames_np.shape[0]):
                pil = Image.fromarray(frames_np[i]).resize((new_w, new_h), Image.LANCZOS)
                resized.append(np.array(pil))
            frames_np = np.stack(resized, axis=0)

        with torch.random.fork_rng(devices=[pipe.device] if str(pipe.device) != "cpu" else []):
            torch.manual_seed(_MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED)
            frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
            frames_tensor = pipe.video_vae.processor.transform_tensor(frames_tensor)
            vae_dtype = next(pipe.video_vae.parameters()).dtype
            frames_tensor = frames_tensor.to(device=pipe.device, dtype=vae_dtype)
            z = pipe.video_vae.encode_base(frames_tensor, process_image=False)  # [1,24,T',H',W']
        z = z.float()

        mean = torch.tensor(_VIDEO_LATENTS_MEAN, device=z.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(_VIDEO_LATENTS_STD, device=z.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        z_norm = (z - mean) / std
        rows = patchify_video(z_norm)
        return rows, int(z.shape[2]), int(z.shape[3]), int(z.shape[4])

    def _encode_audio_ref(self, pipe, waveform, sample_rate: int = _MINIMAX_H3_AUDIO_SAMPLE_RATE):
        """Encode a reference audio waveform per target library.

        Ref: target reference_encoding.py:278-336 `minimax_h3_encode_reference_audio_rows`.

        Input: torch.Tensor [C, L]. If sample_rate != 32000, resamples via torchaudio.
        Ensures stereo (repeat if mono, truncate if >2). Batched encode as [2, 1, L].
        """
        import torchaudio  # local import; needed only for resample

        pipe.load_models_to_device(("audio_vae",))
        model = pipe.audio_vae
        device = next(model.parameters()).device

        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)  # [1,C,L] → [C,L]
        if waveform.dim() != 2:
            raise ValueError(f"expected audio waveform [C, L], got {list(waveform.shape)}")
        waveform = waveform.float()

        if int(sample_rate) != _MINIMAX_H3_AUDIO_SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(
                int(sample_rate), _MINIMAX_H3_AUDIO_SAMPLE_RATE
            )(waveform)

        # Stereo: repeat mono; truncate if >2 channels.
        if waveform.shape[0] < _MINIMAX_H3_AUDIO_CHANNELS:
            repeats = (_MINIMAX_H3_AUDIO_CHANNELS + waveform.shape[0] - 1) // waveform.shape[0]
            waveform = waveform.repeat(repeats, 1)
        # Match the VAE's parameter dtype: VRAM management may hold weights in bf16
        # while the decoded waveform is fp32. (Target upcasts the VAE to fp32 instead;
        # under VRAM management we cannot mutate the wrapped weights, so we cast the input.)
        vae_dtype = next(model.parameters()).dtype
        waveform = waveform[:_MINIMAX_H3_AUDIO_CHANNELS].to(device=device, dtype=vae_dtype)

        with _AudioVAEDeterminismContext():
            # Batched: preprocess([2, 1, L]) → encoder → optional pre_block → mean_proj
            # Ref: target reference_encoding.py:308-317
            audio_data = model.preprocess(waveform.unsqueeze(1), _MINIMAX_H3_AUDIO_SAMPLE_RATE)
            z = model.encoder(audio_data)
            if bool(getattr(model, "attn_proj", False)):
                z = model.pre_block(z.transpose(1, 2)).transpose(1, 2)
            if not hasattr(model, "mean_proj"):
                raise AttributeError(
                    "audio VAE model must expose mean_proj for deterministic mean encoding"
                )
            latent = model.mean_proj(z).float().cpu()  # [2, 32, T] or [2, T, 32]

        if latent.ndim != 3:
            raise ValueError(f"expected 3D audio latent, got {list(latent.shape)}")
        latent_channels = 32
        if int(latent.shape[-1]) != latent_channels:
            if int(latent.shape[1]) != latent_channels:
                raise ValueError(f"cannot canonicalize audio latent {list(latent.shape)}")
            latent = latent.transpose(1, 2).contiguous()  # → [2, T, 32]

        # Normalize per target reference_encoding.py:326-329 (view [1,1,32])
        mean = torch.tensor(_AUDIO_LATENTS_MEAN, dtype=torch.float32).view(1, 1, latent_channels)
        std = torch.tensor(_AUDIO_LATENTS_STD, dtype=torch.float32).view(1, 1, latent_channels)
        normalized = (latent - mean) / std
        rows = normalized.reshape(-1, latent_channels).to(torch.float32).contiguous()
        ref_audio_t = int(latent.shape[1])
        return rows.to(device), ref_audio_t

    def _process_image_ref(self, pipe, img, seed_val):
        """Encode-only pass. Noise aug applied in the coordinating loop."""
        rows, latent_h, latent_w = self._encode_image_ref(pipe, img, seed_val)
        return {
            "kind": "image",
            "clean_rows": rows,
            "latent_h": latent_h, "latent_w": latent_w, "latent_t": 1,
        }

    def _process_video_ref(self, pipe, frames, seed_val, target_frame_count=None):
        rows, latent_t, latent_h, latent_w = self._encode_video_ref(pipe, frames, seed_val, target_frame_count)
        return {
            "kind": "video",
            "clean_rows": rows,
            "latent_t": latent_t, "latent_h": latent_h, "latent_w": latent_w,
        }

    def _process_audio_ref(self, pipe, waveform, sample_rate=_MINIMAX_H3_AUDIO_SAMPLE_RATE):
        rows, ref_audio_t = self._encode_audio_ref(pipe, waveform, sample_rate=sample_rate)
        return {
            "kind": "audio",
            "clean_rows": rows,
            "ref_audio_t": ref_audio_t,
        }

    @staticmethod
    def _frame_count_from_latent_t(latent_t: int) -> int:
        """Invert `latent_t = ((F - 5) // 17) * 5 + 2` for F=17n+5.
        For latent_t >= 2: F = ((latent_t - 2) // 5) * 17 + 5. Ref matches
        target time_request.MINIMAX_H3_SHAPE_PLANNER frame-count alignment.
        """
        latent_t = int(latent_t)
        if latent_t < 2:
            return 5
        return ((latent_t - 2) // 5) * 17 + 5

    def process(self, pipe: MiniMaxH3Pipeline, references, seed, imgvid_cond_noise_aug,
                audio_cond_noise_aug, video_latent_t):
        if references is None:
            return {}
        if not isinstance(references, (list, tuple)) or len(references) == 0:
            return {}

        pipe.load_models_to_device(("video_vae",))
        seed_val = int(seed) if seed is not None else 42
        device = pipe.device
        target_latent_t = int(video_latent_t)
        target_frame_count = self._frame_count_from_latent_t(target_latent_t)

        # First pass: encode each ref, collect clean rows + geometry.
        ref_blocks_out = []
        for ref in references:
            kind = ref["type"]
            if kind == "image":
                block = self._process_image_ref(pipe, ref["data"], seed_val)
            elif kind == "video":
                block = self._process_video_ref(pipe, ref["data"], seed_val, target_frame_count)
            elif kind == "audio":
                data = ref["data"]
                sr = ref.get("sample_rate", _MINIMAX_H3_AUDIO_SAMPLE_RATE)
                block = self._process_audio_ref(pipe, data, sample_rate=sr)
            else:
                raise ValueError(f"unknown reference type: {kind}. supported: image / video / audio")
            ref_blocks_out.append(block)

        # Second pass: apply noise augmentation per target library rules.
        # Ref: condition_noise.py:24-119 (visual) and 122-191 (audio).
        visual_shapes = [
            (int(b["latent_t"]), int(b["latent_h"]), int(b["latent_w"]))
            for b in ref_blocks_out if b["kind"] in ("image", "video")
        ]
        audio_lengths = [
            int(b["ref_audio_t"]) for b in ref_blocks_out if b["kind"] == "audio"
        ]
        imgvid_cond_num_frames = len(visual_shapes)

        # Visual noise aug: per condition, fresh Generator(seed), draw full-target
        # tensor and slice prefix [:latent_t].
        noise_aug = float(imgvid_cond_noise_aug)
        visual_index = 0
        for block in ref_blocks_out:
            if block["kind"] not in ("image", "video"):
                continue
            latent_t_i, latent_h_i, latent_w_i = visual_shapes[visual_index]
            clean = block["clean_rows"].to(device=device, dtype=torch.float32)
            if noise_aug == 1.0:
                anchor = clean
            else:
                full_t = target_latent_t + imgvid_cond_num_frames
                generator = torch.Generator(device="cpu").manual_seed(seed_val)
                noise = torch.randn(
                    1, 24, full_t, latent_h_i, latent_w_i,
                    generator=generator, dtype=torch.float32, device="cpu",
                )[:, :, :latent_t_i]
                noise_rows = patchify_video(noise).to(device=device, dtype=torch.float32)
                ts = torch.tensor(noise_aug, dtype=torch.float32, device=device)
                anchor = ts * clean + (1.0 - ts) * noise_rows
            block["visual_rows"] = anchor
            block["num_visual_rows"] = int(anchor.shape[0])
            del block["clean_rows"]
            visual_index += 1

        # Audio noise aug: per condition, fresh Generator(seed + 1), draw randn of
        # clean_rows shape.
        audio_noise_aug = float(audio_cond_noise_aug)
        for block in ref_blocks_out:
            if block["kind"] != "audio":
                continue
            clean = block["clean_rows"].to(device=device, dtype=torch.float32)
            if audio_noise_aug == 1.0:
                anchor = clean
            else:
                generator = torch.Generator(device="cpu").manual_seed(seed_val + 1)
                noise = torch.randn(
                    clean.shape, generator=generator, dtype=torch.float32, device="cpu",
                ).to(device=device, dtype=torch.float32)
                ts = torch.tensor(audio_noise_aug, dtype=torch.float32, device=device)
                anchor = ts * clean + (1.0 - ts) * noise
            block["audio_rows"] = anchor
            block["num_audio_rows"] = int(anchor.shape[0])
            del block["clean_rows"]

        return {"ref_blocks": ref_blocks_out}


class MiniMaxH3Unit_PackedSequenceBuilder(PipelineUnit):
    _INTERP = 32
    _T_GROUP = 5
    _FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    _FRAME_RESCALE = 5.0 / 3.0
    _SEQ_ALIGN = 64
    _PATCH_H, _PATCH_W = 2, 2
    _TEXT_ID, _AUDIO_FIRST_ID, _AUDIO_ID = -5, -15, -14
    _IMGVID_COND_ID = -11
    _AUDIO_REF_COND_ID = -17
    _VIDEO_FIRST_ID, _VIDEO_ID, _VIDEO_LAST_ID, _PAD_ID = -3, -2, -4, -1

    def __init__(self):
        super().__init__(
            input_params=("prompt_embeds", "video_latent_t", "latent_h", "latent_w", "audio_latent_t",
                          "keyframe_cond_anchor", "keyframe_indices_validated", "ref_blocks"),
            output_params=("packed",),
        )

    @classmethod
    def _axis_from_sqrt_area(cls, dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
        ratio = dim / sqrt_area
        left = (1.0 - ratio) * 0.5
        right = left + ratio
        grid = np.linspace(left, right, dim // patch, endpoint=False) * cls._INTERP
        return torch.from_numpy(grid).to(torch.float64)

    @classmethod
    def _video_t_grid(cls, n: int, origin: float) -> torch.Tensor:
        spans = torch.tensor(
            [cls._FRAME_RESCALE * cls._FRAME_PER_TOKEN[k % cls._T_GROUP] for k in range(n)],
            dtype=torch.float64,
        )
        return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

    @classmethod
    def _temporal_position_span(cls, temporal_length: int) -> float:
        """Temporal position span for patch_t=1, in fp64.

        NOTE: intentionally NOT merged with `_video_t_span`. This variant sums
        via numpy (pairwise summation), matching the fl2va anchor computation,
        while `_video_t_span` sums sequentially, matching the ref2va
        t-origin accumulation. The two orders diverge in the last ulp from
        n=16 onward, so each path must keep its own summation order.

        Ref: target packed_sequence.py:112-124.
        """
        spans = np.ones(int(temporal_length), dtype=np.float64) * cls._FRAME_RESCALE
        for token_index in range(cls._T_GROUP):
            spans[token_index::cls._T_GROUP] *= cls._FRAME_PER_TOKEN[token_index]
        return float(spans.sum())

    @classmethod
    def _video_t_span(cls, n: int) -> float:
        # Sequential fp64 summation on purpose — see _temporal_position_span for
        # why the two span implementations must not be unified.
        # Ref: target packed_sequence.py:290-293.
        return sum(cls._FRAME_RESCALE * cls._FRAME_PER_TOKEN[k % cls._T_GROUP] for k in range(n))

    @classmethod
    def _build_packed_t2va(cls, text_len, latent_t, latent_h, latent_w, audio_t, audio_channel=2):
        """t2va layout: [text | audio | video | pad]."""
        ph, pw = latent_h // cls._PATCH_H, latent_w // cls._PATCH_W
        frame_rows = ph * pw
        video_rows = latent_t * frame_rows
        audio_rows = audio_t * audio_channel
        used = text_len + audio_rows + video_rows
        seq_len = ((used + cls._SEQ_ALIGN - 1) // cls._SEQ_ALIGN) * cls._SEQ_ALIGN

        text_sl = slice(0, text_len)
        audio_sl = slice(text_len, text_len + audio_rows)
        video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)

        input_ids = torch.full((seq_len,), cls._PAD_ID, dtype=torch.int64)
        input_ids[text_sl] = cls._TEXT_ID
        input_ids[audio_sl] = cls._AUDIO_ID
        input_ids[audio_sl.start] = cls._AUDIO_FIRST_ID
        input_ids[video_sl] = cls._VIDEO_ID
        input_ids[video_sl.start] = cls._VIDEO_FIRST_ID
        input_ids[video_sl.stop - 1] = cls._VIDEO_LAST_ID

        img_pos = torch.arange(video_sl.start, video_sl.stop)
        update_mask = torch.ones(img_pos.shape[0], dtype=torch.bool)
        audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
        text_pos = torch.arange(0, text_len)

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
        t_grid = cls._video_t_grid(latent_t, float(text_len))
        sqrt_area = np.sqrt(latent_h * latent_w)
        h_grid = cls._axis_from_sqrt_area(latent_h, cls._PATCH_H, sqrt_area)
        w_grid = cls._axis_from_sqrt_area(latent_w, cls._PATCH_W, sqrt_area)
        hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
        frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
        video_g = torch.empty(latent_t, frame_rows, 3, dtype=torch.float64)
        video_g[:, :, 0] = t_grid[:, None]
        video_g[:, :, 1:] = frame[None]
        g[video_sl] = video_g.reshape(-1, 3)
        audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
        g[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
        g[audio_sl, 2] = torch.cat([
            torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64),
            torch.full((audio_rows - audio_t,), float(w_grid[-1]), dtype=torch.float64),
        ])

        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[text_sl] = 1
        token_tags[audio_sl] = 2
        token_tags[img_pos] = 0

        cu = torch.tensor([0, used, seq_len], dtype=torch.int32)
        return {
            "seq_len": int(seq_len),
            "img_pos": img_pos,
            "audio_pos": audio_pos,
            "text_pos": text_pos,
            "update_mask": update_mask,
            "img_position_ids": g,
            "token_tags": token_tags,
            "cu_seqlens": cu,
            "text_len": int(text_len),
            "audio_channel": audio_channel,
            "audio_t": audio_t,
            "latent_t": latent_t,
            "latent_h_patched": ph,
            "latent_w_patched": pw,
        }

    @classmethod
    def _build_packed_fl2va(cls, text_len, latent_t, latent_h, latent_w, audio_t,
                            keyframe_indices, audio_channel=2):
        """fl2va layout: [text | cond | audio | video | pad]."""
        ph, pw = latent_h // cls._PATCH_H, latent_w // cls._PATCH_W
        frame_rows = ph * pw
        video_rows = latent_t * frame_rows
        audio_rows = audio_t * audio_channel
        num_keyframes = len(keyframe_indices)
        cond_rows = num_keyframes * frame_rows
        used = text_len + cond_rows + audio_rows + video_rows
        seq_len = ((used + cls._SEQ_ALIGN - 1) // cls._SEQ_ALIGN) * cls._SEQ_ALIGN

        text_sl = slice(0, text_len)
        cond_sl = slice(text_len, text_len + cond_rows)
        audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
        video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)

        input_ids = torch.full((seq_len,), cls._PAD_ID, dtype=torch.int64)
        input_ids[text_sl] = cls._TEXT_ID
        input_ids[cond_sl] = cls._IMGVID_COND_ID
        input_ids[audio_sl] = cls._AUDIO_ID
        input_ids[audio_sl.start] = cls._AUDIO_FIRST_ID
        input_ids[video_sl] = cls._VIDEO_ID
        input_ids[video_sl.start] = cls._VIDEO_FIRST_ID
        input_ids[video_sl.stop - 1] = cls._VIDEO_LAST_ID

        # img_pos covers both cond AND video rows; update_mask separates them
        img_pos = torch.cat([
            torch.arange(cond_sl.start, cond_sl.stop),
            torch.arange(video_sl.start, video_sl.stop),
        ])
        update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
        update_mask[cond_rows:] = True

        audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
        text_pos = torch.arange(0, text_len)

        # RoPE position grid
        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

        sqrt_area = np.sqrt(latent_h * latent_w)
        h_grid = cls._axis_from_sqrt_area(latent_h, cls._PATCH_H, sqrt_area)
        w_grid = cls._axis_from_sqrt_area(latent_w, cls._PATCH_W, sqrt_area)
        hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
        frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

        # Condition rows: temporal position depends on frame_index
        t_grid_video = cls._video_t_grid(latent_t, float(text_len))
        temporal_span = cls._temporal_position_span(latent_t)
        for i, idx in enumerate(keyframe_indices):
            sl = slice(i * frame_rows, (i + 1) * frame_rows)
            if idx == 0:
                cond_t = float(text_len)
            else:  # idx == -1
                cond_t = float(text_len) + temporal_span - cls._FRAME_RESCALE
            cond_g = torch.empty(frame_rows, 3, dtype=torch.float64)
            cond_g[:, 0] = cond_t
            cond_g[:, 1:] = frame
            g[cond_sl.start + sl.start: cond_sl.start + sl.stop] = cond_g

        # Video target rows
        video_g = torch.empty(latent_t, frame_rows, 3, dtype=torch.float64)
        video_g[:, :, 0] = t_grid_video[:, None]
        video_g[:, :, 1:] = frame[None]
        g[video_sl] = video_g.reshape(-1, 3)

        # Audio rows
        audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
        g[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
        g[audio_sl, 2] = torch.cat([
            torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64),
            torch.full((audio_rows - audio_t,), float(w_grid[-1]), dtype=torch.float64),
        ])

        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[text_sl] = 1
        token_tags[audio_sl] = 2
        token_tags[img_pos] = 0  # both cond and video rows are tagged as video (0)

        cu = torch.tensor([0, used, seq_len], dtype=torch.int32)
        return {
            "seq_len": int(seq_len),
            "img_pos": img_pos,
            "audio_pos": audio_pos,
            "text_pos": text_pos,
            "update_mask": update_mask,
            "img_position_ids": g,
            "token_tags": token_tags,
            "cu_seqlens": cu,
            "text_len": int(text_len),
            "audio_channel": audio_channel,
            "audio_t": audio_t,
            "latent_t": latent_t,
            "latent_h_patched": ph,
            "latent_w_patched": pw,
            "cond_rows": cond_rows,
        }

    @classmethod
    def _build_packed_ref2va(cls, text_len, latent_t, latent_h, latent_w, audio_t, ref_blocks, audio_channel=2):
        """ref2va layout: [text | ref_0 | ref_1 | ... | target_audio | target_video | pad].
        Each ref block uses its OWN spatial grid; temporal positions accumulate
        via t_cursor. Returns packed dict with additional keys: audio_update_mask,
        ref_visual_rows_count, ref_audio_rows_count."""
        ph = latent_h // cls._PATCH_H
        pw = latent_w // cls._PATCH_W
        target_frame_rows = ph * pw
        target_video_rows = latent_t * target_frame_rows
        target_audio_rows = audio_t * audio_channel

        # First pass: compute per-block dims and total ref rows to size the seq.
        block_dims = []  # each: {"kind", "visual_rows", "audio_rows", "latent_t", "latent_h", "latent_w", "ref_audio_t"}
        total_ref_visual_rows = 0
        total_ref_audio_rows = 0
        for b in ref_blocks:
            kind = b["kind"]
            info = {"kind": kind}
            if kind == "image":
                lh_r = b["latent_h"]
                lw_r = b["latent_w"]
                v_rows = (lh_r // cls._PATCH_H) * (lw_r // cls._PATCH_W)
                info["visual_rows"] = v_rows
                info["audio_rows"] = 0
                info["latent_t"] = 1
                info["latent_h"] = lh_r
                info["latent_w"] = lw_r
                total_ref_visual_rows += v_rows
            elif kind == "video":
                lt_r = b["latent_t"]
                lh_r = b["latent_h"]
                lw_r = b["latent_w"]
                v_rows = lt_r * (lh_r // cls._PATCH_H) * (lw_r // cls._PATCH_W)
                info["visual_rows"] = v_rows
                info["audio_rows"] = 0
                info["latent_t"] = lt_r
                info["latent_h"] = lh_r
                info["latent_w"] = lw_r
                total_ref_visual_rows += v_rows
            elif kind == "audio":
                a_rows = b["ref_audio_t"] * audio_channel
                info["visual_rows"] = 0
                info["audio_rows"] = a_rows
                info["ref_audio_t"] = b["ref_audio_t"]
                total_ref_audio_rows += a_rows
            else:
                raise ValueError(f"unknown ref kind: {kind}")
            block_dims.append(info)

        used = text_len + total_ref_visual_rows + total_ref_audio_rows + target_audio_rows + target_video_rows
        seq_len = ((used + cls._SEQ_ALIGN - 1) // cls._SEQ_ALIGN) * cls._SEQ_ALIGN

        input_ids = torch.full((seq_len,), cls._PAD_ID, dtype=torch.int64)
        input_ids[0:text_len] = cls._TEXT_ID

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[0:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[0:text_len] = 1

        # Iterate through ref blocks, placing them contiguously.
        cursor = text_len
        t_cursor = float(text_len)
        ref_visual_pos_parts = []
        ref_audio_pos_parts = []
        # Target w_grid used for audio W-axis (channel separation)
        target_sqrt_area = float(np.sqrt(latent_h * latent_w))
        target_w_grid = cls._axis_from_sqrt_area(latent_w, cls._PATCH_W, target_sqrt_area)

        for info in block_dims:
            kind = info["kind"]
            if kind == "image":
                v_rows = info["visual_rows"]
                lh_r = info["latent_h"]
                lw_r = info["latent_w"]
                r_ph = lh_r // cls._PATCH_H
                r_pw = lw_r // cls._PATCH_W
                # Own spatial grid
                sqrt_area = float(np.sqrt(lh_r * lw_r))
                h_grid = cls._axis_from_sqrt_area(lh_r, cls._PATCH_H, sqrt_area)
                w_grid = cls._axis_from_sqrt_area(lw_r, cls._PATCH_W, sqrt_area)
                hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
                frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
                sl = slice(cursor, cursor + v_rows)
                input_ids[sl] = cls._IMGVID_COND_ID
                g[sl, 0] = t_cursor
                g[sl, 1:] = frame
                token_tags[sl] = 0
                ref_visual_pos_parts.append(torch.arange(sl.start, sl.stop))
                cursor += v_rows
                t_cursor += 1.0

            elif kind == "video":
                v_rows = info["visual_rows"]
                lt_r = info["latent_t"]
                lh_r = info["latent_h"]
                lw_r = info["latent_w"]
                r_ph = lh_r // cls._PATCH_H
                r_pw = lw_r // cls._PATCH_W
                frame_rows_r = r_ph * r_pw
                sqrt_area = float(np.sqrt(lh_r * lw_r))
                h_grid = cls._axis_from_sqrt_area(lh_r, cls._PATCH_H, sqrt_area)
                w_grid = cls._axis_from_sqrt_area(lw_r, cls._PATCH_W, sqrt_area)
                hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
                frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
                t_grid = cls._video_t_grid(lt_r, t_cursor)
                sl = slice(cursor, cursor + v_rows)
                input_ids[sl] = cls._IMGVID_COND_ID
                video_g = torch.empty(lt_r, frame_rows_r, 3, dtype=torch.float64)
                video_g[:, :, 0] = t_grid[:, None]
                video_g[:, :, 1:] = frame[None]
                g[sl] = video_g.reshape(-1, 3)
                token_tags[sl] = 0
                ref_visual_pos_parts.append(torch.arange(sl.start, sl.stop))
                cursor += v_rows
                t_cursor += cls._video_t_span(lt_r)

            elif kind == "audio":
                a_rows = info["audio_rows"]
                ref_at = info["ref_audio_t"]
                sl = slice(cursor, cursor + a_rows)
                input_ids[sl] = cls._AUDIO_REF_COND_ID
                # Temporal: (t_cursor + arange(ref_at)).repeat(audio_channel)
                a_t_grid = t_cursor + torch.arange(ref_at, dtype=torch.float64)
                g[sl, 0] = a_t_grid.repeat(audio_channel)
                # W-axis: channel-separated
                g[sl, 2] = torch.cat([
                    torch.full((ref_at,), float(target_w_grid[0]), dtype=torch.float64),
                    torch.full((a_rows - ref_at,), float(target_w_grid[-1]), dtype=torch.float64),
                ])
                token_tags[sl] = 2
                ref_audio_pos_parts.append(torch.arange(sl.start, sl.stop))
                cursor += a_rows
                t_cursor += float(ref_at)

        # Target audio + target video after ref blocks
        target_audio_sl = slice(cursor, cursor + target_audio_rows)
        target_video_sl = slice(target_audio_sl.stop, target_audio_sl.stop + target_video_rows)

        input_ids[target_audio_sl] = cls._AUDIO_ID
        input_ids[target_audio_sl.start] = cls._AUDIO_FIRST_ID
        input_ids[target_video_sl] = cls._VIDEO_ID
        input_ids[target_video_sl.start] = cls._VIDEO_FIRST_ID
        input_ids[target_video_sl.stop - 1] = cls._VIDEO_LAST_ID

        # Target spatial grid (own)
        h_grid_t = cls._axis_from_sqrt_area(latent_h, cls._PATCH_H, target_sqrt_area)
        w_grid_t = cls._axis_from_sqrt_area(latent_w, cls._PATCH_W, target_sqrt_area)
        hh_t, ww_t = torch.meshgrid(h_grid_t, w_grid_t, indexing="ij")
        frame_t = torch.stack([hh_t.reshape(-1), ww_t.reshape(-1)], dim=-1)
        t_grid_v = cls._video_t_grid(latent_t, t_cursor)
        video_g_t = torch.empty(latent_t, target_frame_rows, 3, dtype=torch.float64)
        video_g_t[:, :, 0] = t_grid_v[:, None]
        video_g_t[:, :, 1:] = frame_t[None]
        g[target_video_sl] = video_g_t.reshape(-1, 3)

        target_audio_t_grid = t_cursor + torch.arange(audio_t, dtype=torch.float64)
        g[target_audio_sl, 0] = target_audio_t_grid.repeat(audio_channel)
        g[target_audio_sl, 2] = torch.cat([
            torch.full((audio_t,), float(w_grid_t[0]), dtype=torch.float64),
            torch.full((target_audio_rows - audio_t,), float(w_grid_t[-1]), dtype=torch.float64),
        ])

        # Build combined img_pos / audio_pos
        target_video_pos = torch.arange(target_video_sl.start, target_video_sl.stop)
        target_audio_pos = torch.arange(target_audio_sl.start, target_audio_sl.stop)

        if ref_visual_pos_parts:
            img_pos = torch.cat(ref_visual_pos_parts + [target_video_pos])
        else:
            img_pos = target_video_pos
        if ref_audio_pos_parts:
            audio_pos = torch.cat(ref_audio_pos_parts + [target_audio_pos])
        else:
            audio_pos = target_audio_pos

        update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
        update_mask[total_ref_visual_rows:] = True
        audio_update_mask = torch.zeros(audio_pos.shape[0], dtype=torch.bool)
        audio_update_mask[total_ref_audio_rows:] = True

        token_tags[img_pos] = 0
        token_tags[target_audio_sl] = 2

        text_pos = torch.arange(0, text_len)
        cu = torch.tensor([0, used, seq_len], dtype=torch.int32)

        return {
            "seq_len": int(seq_len),
            "img_pos": img_pos,
            "audio_pos": audio_pos,
            "text_pos": text_pos,
            "update_mask": update_mask,
            "audio_update_mask": audio_update_mask,
            "img_position_ids": g,
            "token_tags": token_tags,
            "cu_seqlens": cu,
            "text_len": int(text_len),
            "audio_channel": audio_channel,
            "audio_t": audio_t,
            "latent_t": latent_t,
            "latent_h_patched": ph,
            "latent_w_patched": pw,
            "cond_rows": total_ref_visual_rows,  # visual ref rows count
            "ref_audio_rows": total_ref_audio_rows,
        }

    def process(self, pipe: MiniMaxH3Pipeline, prompt_embeds, video_latent_t, latent_h, latent_w, audio_latent_t,
                keyframe_cond_anchor=None, keyframe_indices_validated=None, ref_blocks=None):
        text_len = int(prompt_embeds.shape[0])
        if ref_blocks is not None:
            packed = self._build_packed_ref2va(
                text_len, video_latent_t, latent_h, latent_w, audio_latent_t, ref_blocks,
            )
        elif keyframe_cond_anchor is not None:
            packed = self._build_packed_fl2va(
                text_len, video_latent_t, latent_h, latent_w, audio_latent_t,
                keyframe_indices_validated,
            )
        else:
            packed = self._build_packed_t2va(text_len, video_latent_t, latent_h, latent_w, audio_latent_t)

        dev = pipe.device
        packed["img_pos"] = packed["img_pos"].to(dev, torch.long)
        packed["audio_pos"] = packed["audio_pos"].to(dev, torch.long)
        packed["text_pos"] = packed["text_pos"].to(dev, torch.long)
        packed["update_mask"] = packed["update_mask"].to(dev)
        if "audio_update_mask" in packed:
            packed["audio_update_mask"] = packed["audio_update_mask"].to(dev)
        packed["img_position_ids"] = packed["img_position_ids"][None].to(dev)  # [1,S,3] fp64
        packed["token_tags"] = packed["token_tags"].to(dev, torch.long)
        packed["cu_seqlens"] = packed["cu_seqlens"].to(dev, torch.int32)
        return {"packed": packed}


def model_fn_minimax_h3(
    dit, video_latents, audio_latents, packed, prompt_embeds,
    t_video, t_audio, device, torch_dtype,
    keyframe_cond_anchor=None, imgvid_cond_noise_aug=0.999,
    ref_blocks=None, audio_cond_noise_aug=1.0,
    use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
    **kwargs,
):
    # ---- patchify (in): natural latents -> packed rows ----
    video_rows = patchify_video(video_latents.to(device, torch.float32))
    audio_rows = pack_audio(audio_latents.to(device, torch.float32))

    seq_len = packed["seq_len"]
    img_pos = packed["img_pos"]
    audio_pos = packed["audio_pos"]
    text_pos = packed["text_pos"]
    cu = packed["cu_seqlens"]
    text_len = packed["text_len"]
    update_mask = packed["update_mask"]
    cond_rows_count = packed.get("cond_rows", 0)
    ref_audio_rows_count = packed.get("ref_audio_rows", 0)

    x = torch.zeros(1, seq_len, 96, dtype=torch.float32, device=device)
    audio_x = torch.zeros(1, seq_len, 32, dtype=torch.float32, device=device)

    # Collect ref anchor rows (for Ref2AV) or use keyframe anchor (for FL2AV)
    ref_visual_anchor = None
    ref_audio_anchor = None
    if ref_blocks is not None:
        visual_parts = []
        audio_parts = []
        for b in ref_blocks:
            if b.get("visual_rows") is not None:
                visual_parts.append(b["visual_rows"])
            if b.get("audio_rows") is not None:
                audio_parts.append(b["audio_rows"])
        if visual_parts:
            ref_visual_anchor = torch.cat(visual_parts, dim=0).to(device, torch.float32)
        if audio_parts:
            ref_audio_anchor = torch.cat(audio_parts, dim=0).to(device, torch.float32)

    if ref_visual_anchor is not None and cond_rows_count > 0:
        # Ref2AV visual layout: img_pos[:cond_rows] = ref visual, img_pos[cond_rows:] = target video
        ref_pos = img_pos[:cond_rows_count]
        target_pos = img_pos[cond_rows_count:]
        x[0].index_copy_(0, target_pos, video_rows)
        x[0].index_copy_(0, ref_pos, ref_visual_anchor)
    elif keyframe_cond_anchor is not None and cond_rows_count > 0:
        # FL2AV: same visual layout with keyframe anchor
        cond_pos = img_pos[:cond_rows_count]
        target_pos = img_pos[cond_rows_count:]
        x[0].index_copy_(0, target_pos, video_rows)
        x[0].index_copy_(0, cond_pos, keyframe_cond_anchor.to(device, torch.float32))
    else:
        # t2va: all img_pos are video target
        x[0].index_copy_(0, img_pos, video_rows)

    # Audio: ref audio (if any) at audio_pos[:ref_audio_rows], target at audio_pos[ref_audio_rows:]
    if ref_audio_anchor is not None and ref_audio_rows_count > 0:
        ref_audio_pos = audio_pos[:ref_audio_rows_count]
        target_audio_pos = audio_pos[ref_audio_rows_count:]
        audio_x[0].index_copy_(0, target_audio_pos, audio_rows)
        audio_x[0].index_copy_(0, ref_audio_pos, ref_audio_anchor)
    else:
        audio_x[0].index_copy_(0, audio_pos, audio_rows)

    # Timesteps: target video rows get t_video, target audio rows get t_audio.
    # Ref/cond visual rows get max(t_video, imgvid_cond_noise_aug).
    # Ref audio rows get max(t_audio, audio_cond_noise_aug) (default 1.0 → always 1.0).
    timesteps = torch.full((seq_len,), float(t_video), dtype=torch.float32, device=device)
    timesteps[audio_pos] = float(t_audio)
    if cond_rows_count > 0:
        cond_t = max(float(t_video), float(imgvid_cond_noise_aug))
        timesteps[img_pos[:cond_rows_count]] = cond_t
    if ref_audio_rows_count > 0:
        audio_ref_t = max(float(t_audio), float(audio_cond_noise_aug))
        timesteps[audio_pos[:ref_audio_rows_count]] = audio_ref_t

    unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)

    refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)
    # Target library passes FULL update_mask + FULL img_pos_for_infer_output_info to the
    # DiT (ref denoise_loop.py:82-102). The DiT internally gathers via infer_out_pos and
    # zeros condition rows via update_mask (ref minimax_h3.py:1012-1036). We slice the
    # target-only rows out AFTER the forward call before unpatchify.
    v_video_rows, v_audio_rows = dit(
        x=x,
        audio_x=audio_x,
        img_position_ids=packed["img_position_ids"],
        unique_timesteps=unique_timesteps,
        inverse_indices=inverse_indices,
        update_mask=update_mask,
        token_tags=packed["token_tags"],
        prompt_embeds=prompt_embeds.to(device, torch.bfloat16),
        img_pos_info={"position_ids": img_pos},
        audio_pos_info={"position_ids": audio_pos},
        text_pos_info={"position_ids": text_pos},
        img_pos_for_infer_output_info={"position_ids": img_pos},
        packed_seq_params={"cu_seqlens_q": cu, "max_seqlen_q": int(cu[1])},
        refiner_packed_seq_params={"cu_seqlens_q": refiner_cu, "max_seqlen_q": text_len},
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        skip_mask_out_condition=False,
    )

    # Slice target-only rows for unpatchify (cond/ref rows have been masked to 0 by DiT).
    if cond_rows_count > 0:
        v_video_rows = v_video_rows[cond_rows_count:]
    if ref_audio_rows_count > 0:
        v_audio_rows = v_audio_rows[ref_audio_rows_count:]

    # ---- unpatchify (out) + velocity negation ----
    v_video = unpatchify_video(v_video_rows.float(), packed["latent_t"], packed["latent_h_patched"], packed["latent_w_patched"])
    v_audio = unpack_audio(v_audio_rows.float(), packed["audio_channel"], packed["audio_t"])
    return -v_video, -v_audio
