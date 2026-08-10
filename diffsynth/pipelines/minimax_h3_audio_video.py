import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.minimax_h3_dit import MiniMaxH3DiT, patchify_video, unpatchify_video, pack_audio, unpack_audio
from ..models.minimax_h3_text_encoder import MiniMaxH3TextEncoder, presentation_t2va, presentation_fl2va, presentation_ref2va, sample_qwen_video_frames, image_token_counts, video_token_counts
from ..models.minimax_h3_video_vae import MiniMaxH3VideoVAE
from ..models.minimax_h3_audio_vae import MiniMaxH3AudioVAE
from ..models.minimax_constant import *

class MiniMaxH3Pipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=32, width_division_factor=32, time_division_factor=17, time_division_remainder=5)
        self.scheduler = FlowMatchScheduler("MiniMax-H3")
        self.scheduler_audio = FlowMatchScheduler("MiniMax-H3")
        self.text_encoder: MiniMaxH3TextEncoder = None
        self.dit: MiniMaxH3DiT = None
        self.video_vae: MiniMaxH3VideoVAE = None
        self.audio_vae: MiniMaxH3AudioVAE = None
        self.tokenizer = None
        self.processor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            MiniMaxH3Unit_ShapeChecker(),
            MiniMaxH3Unit_NoiseInitializer(),
            MiniMaxH3Unit_InputVideoEmbedder(),
            MiniMaxH3Unit_InputAudioEmbedder(),
            MiniMaxH3Unit_KeyframeEncoder(),
            MiniMaxH3Unit_ReferenceEncoder(),
            MiniMaxH3Unit_PromptEmbedder(),
            MiniMaxH3Unit_PackedSequenceBuilder(),
        ]
        self.model_fn = model_fn_minimax_h3
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
        vram_limit: float = None,
    ):
        pipe = MiniMaxH3Pipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("minimax_h3_text_encoder")
        pipe.dit = model_pool.fetch_model("minimax_h3_dit")
        pipe.video_vae = model_pool.fetch_model("minimax_h3_video_vae")
        pipe.audio_vae = model_pool.fetch_model("minimax_h3_audio_vae")
        if processor_config is not None:
            processor_config.download_if_necessary()
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
            pipe.tokenizer = pipe.processor.tokenizer
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = " ",
        height: int = 768,
        width: int = 1344,
        num_frames: int = 124,
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
        # Keyframe to Video
        keyframes=None,
        keyframe_indices=None,
        # Reference to Video
        references=None,
        progress_bar_cmd=tqdm,
    ):
        """Generate a joint video + audio sample.

        `num_frames` is snapped up to the nearest 17n+5 and that aligned value
        drives every downstream shape (video/audio latent lengths, reference video
        capping), so the returned clip may be slightly longer than requested.

        `keyframes` (FL2AV) is a list of PIL images with `keyframe_indices` in
        {0, -1}; both are resized onto the target canvas.

        `references` (Ref2AV) is a list of dicts in request order:

            {"type": "image",       "image": PIL.Image}
            {"type": "video",       "video": list[PIL.Image]}   # silent
            {"type": "audio",       "audio": Tensor[C, L], "sample_rate": int}
            {"type": "video_audio", "video": list[PIL.Image],
                                    "audio": Tensor[C, L], "sample_rate": int}

        Input contract: `video` frame lists must ALREADY be 24fps the pipeline never resamples frame rate.
        """
        self.scheduler.set_timesteps(num_inference_steps, shift=flow_shift)
        self.scheduler_audio.set_timesteps(num_inference_steps, shift=audio_flow_shift)

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width, "num_frames": num_frames,
            "seed": seed, "rand_device": rand_device,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": use_gradient_checkpointing_offload,
            "keyframes": keyframes,
            "keyframe_indices": keyframe_indices,
            "references": references,
        }

        # 3. Unit chain
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
                **models, t_video=t_video, t_audio=t_audio, device=self.device,
            )
            inputs_shared["video_latents"] = self.step(self.scheduler, inputs_shared["video_latents"], progress_id, noise_pred=noise_pred_video)
            inputs_shared["audio_latents"] = self.step(self.scheduler_audio, inputs_shared["audio_latents"], progress_id, noise_pred=noise_pred_audio)

        # 5. Decode
        self.load_models_to_device(["video_vae"])
        frames = self.video_vae.decode_video(inputs_shared["video_latents"], dtype=self.torch_dtype, tiled=tiled, tile_size=tile_size, tile_overlap=tile_overlap)
        video = self.vae_output_to_video(frames, min_value=0, max_value=1)

        self.load_models_to_device(["audio_vae"])
        waveform = self.audio_vae.decode_audio(inputs_shared["audio_latents"], dtype=self.torch_dtype)
        audio = self.output_audio_format_check(waveform)
        return video, audio


class MiniMaxH3Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: MiniMaxH3Pipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class MiniMaxH3Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("seed", "num_frames", "height", "width", "rand_device"),
            output_params=("video_latents", "audio_latents"),
        )

    def process(self, pipe: MiniMaxH3Pipeline, seed, num_frames, height, width, rand_device):
        video_latent_t, latent_h, latent_w = ((num_frames - 5) // 17) * 5 + 2, height // 16, width // 16
        video_latents = pipe.generate_noise((1, 24, video_latent_t, latent_h, latent_w), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        audio_latent_t = round(num_frames / 24.0 * 40.0)
        audio_latents = pipe.generate_noise((2, 32, audio_latent_t), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"video_latents": video_latents, "audio_latents": audio_latents}


class MiniMaxH3Unit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("keyframes", "ref_blocks", "height", "width"),
            output_params=("prompt_embeds", "text_token_tags"),
            onload_model_names=("text_encoder",),
        )

    def preprocess_ref_blocks(self, pipe: MiniMaxH3Pipeline, prompt, ref_blocks):
        pixel_values = image_grid_thw = pixel_values_videos = video_grid_thw = None
        counters = {"image": 0, "audio": 0, "video": 0}
        images, videos, timestamps_per_video, condition_labels = [], [], [], []
        for block in ref_blocks:
            kind = block["kind"]
            if kind == "image":
                counters["image"] += 1
                condition_labels.append(("image", counters["image"]))
                images.append(block["prepared_image"])
            elif kind == "audio":
                counters["audio"] += 1
                condition_labels.append(("audio", counters["audio"]))
            elif kind in ("video", "video_audio"):
                if int(block["ref_audio_t"]) > 0:
                    counters["audio"] += 1
                    condition_labels.append(("audio", counters["audio"]))
                counters["video"] += 1
                condition_labels.append(("video", counters["video"]))
                sampled, timestamps = sample_qwen_video_frames(block["prepared_frames"])
                videos.append(np.stack([np.asarray(f) for f in sampled]))
                timestamps_per_video.append(timestamps)
            else:
                raise ValueError(f"unknown reference kind: {kind}")

        image_counts, video_counts, video_timestamps = [], [], []
        if len(images) > 0:
            pixel_values, image_grid_thw, image_counts = image_token_counts(pipe.processor, images)
        if len(videos) > 0:
            pixel_values_videos, video_grid_thw, video_counts, video_timestamps = video_token_counts(pipe.processor, videos, timestamps_per_video)
        input_ids, text_token_tags = presentation_ref2va(pipe.tokenizer, prompt, condition_labels, image_counts, video_counts, video_timestamps)
        return input_ids, text_token_tags, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw

    def process(self, pipe: MiniMaxH3Pipeline, prompt, keyframes=None, ref_blocks=None, height=None, width=None):
        pipe.load_models_to_device(("text_encoder",))
        pixel_values = image_grid_thw = pixel_values_videos = video_grid_thw = None
        if ref_blocks:
            input_ids, text_token_tags, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = self.preprocess_ref_blocks(pipe, prompt, ref_blocks)
        elif keyframes:
            keyframes = [img.convert("RGB").resize((width, height)) for img in keyframes]
            pixel_values, image_grid_thw, image_counts = image_token_counts(pipe.processor, keyframes)
            input_ids, text_token_tags = presentation_fl2va(pipe.tokenizer, prompt, image_counts)
        else:
            input_ids, text_token_tags = presentation_t2va(pipe.tokenizer, prompt)

        ids = input_ids.unsqueeze(0).to(pipe.device)
        kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values.to(pipe.device, pipe.torch_dtype)
            kwargs["image_grid_thw"] = image_grid_thw.to(pipe.device, torch.long)
        if pixel_values_videos is not None:
            kwargs["pixel_values_videos"] = pixel_values_videos.to(pipe.device, pipe.torch_dtype)
            kwargs["video_grid_thw"] = video_grid_thw.to(pipe.device, torch.long)
        hidden = pipe.text_encoder(**kwargs)

        return {"prompt_embeds": hidden.to(pipe.device, pipe.torch_dtype), "text_token_tags": text_token_tags.to(pipe.device, torch.long)}


def _encode_audio_waveform(pipe: "MiniMaxH3Pipeline", waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    import torchaudio  # local import; needed only for resample

    pipe.load_models_to_device(("audio_vae",))

    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)  # [1,C,L] -> [C,L]
    if waveform.dim() != 2:
        raise ValueError(f"expected audio waveform [C, L], got {list(waveform.shape)}")
    waveform = waveform.float()

    if int(sample_rate) != pipe.audio_vae.sample_rate:
        waveform = torchaudio.transforms.Resample(int(sample_rate), pipe.audio_vae.sample_rate)(waveform)

    if waveform.shape[0] < AUDIO_CHANNELS:
        repeats = (AUDIO_CHANNELS + waveform.shape[0] - 1) // waveform.shape[0]
        waveform = waveform.repeat(repeats, 1)

    return pipe.audio_vae.encode_audio(
        waveform[:AUDIO_CHANNELS].to(pipe.device), dtype=pipe.torch_dtype,
    )  # [C, 32, T]


class MiniMaxH3Unit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video",),
            output_params=("video_latents", "input_latents"),
            onload_model_names=("video_vae",)
        )

    def process(self, pipe: MiniMaxH3Pipeline, input_video):
        if input_video is None or not pipe.scheduler.training:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        used = _trim_reference_video_length(pipe, len(input_video))
        frames_np = np.stack([np.asarray(f.convert("RGB")) for f in input_video[:used]], axis=0)
        frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        latents = pipe.video_vae.encode_video(frames_tensor.to(pipe.device), dtype=pipe.torch_dtype).to(pipe.torch_dtype)
        return {"video_latents": latents, "input_latents": latents}


class MiniMaxH3Unit_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_audio",),
            output_params=("audio_latents", "audio_input_latents"),
            onload_model_names=("audio_vae",)
        )

    def process(self, pipe: MiniMaxH3Pipeline, input_audio):
        if input_audio is None or not pipe.scheduler.training:
            return {}
        waveform, sample_rate = input_audio
        latents = _encode_audio_waveform(pipe, waveform, int(sample_rate)).to(pipe.torch_dtype)  # [C,32,T]
        return {"audio_latents": latents, "audio_input_latents": latents}


class MiniMaxH3Unit_KeyframeEncoder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("keyframes", "keyframe_indices", "video_latents", "rand_device", "seed", "height", "width"),
            output_params=("keyframe_cond_anchor", "keyframe_indices_validated"),
            onload_model_names=("video_vae",)
        )

    def process(self, pipe: MiniMaxH3Pipeline, keyframes, keyframe_indices, video_latents, rand_device, seed, height, width):
        if keyframes is None:
            return {}
        video_latent_t, latent_h, latent_w = (int(x) for x in video_latents.shape[2:])
        if keyframe_indices is None:
            raise ValueError("keyframe_indices must be provided when keyframes is not None")
        if len(keyframes) != len(keyframe_indices):
            raise ValueError(f"len(keyframes)={len(keyframes)} != len(keyframe_indices)={len(keyframe_indices)}")
        for idx in keyframe_indices:
            if idx not in (0, -1):
                raise ValueError(f"keyframe_indices must be 0 or -1, got {idx}")

        pipe.load_models_to_device(("video_vae",))
        device = pipe.device

        all_cond_rows = []
        for img in keyframes:
            img_tensor = pipe.preprocess_image(img.convert("RGB").resize((width, height)), min_value=0)
            z_norm = pipe.video_vae.encode_video(img_tensor, process_image=True)  # [1,24,1,H',W']
            rows = patchify_video(z_norm)
            all_cond_rows.append(rows)

        clean_cond_rows = torch.cat(all_cond_rows, dim=0).to(device=device, dtype=pipe.torch_dtype)

        seed_val = int(seed) if seed is not None else 42
        num_cond_frames = len(keyframes)
        noise_aug = IMGVID_COND_NOISE_AUG
        if noise_aug == 1.0:
            cond_anchor = clean_cond_rows
        else:
            frame_rows = (latent_h // 2) * (latent_w // 2)
            timestep_tensor = torch.tensor(noise_aug, dtype=pipe.torch_dtype, device=device)
            parts = []
            for i in range(num_cond_frames):
                latent_t_i = 1  # image keyframe = single-frame latent
                full_t = int(video_latent_t) + num_cond_frames
                noise = pipe.generate_noise((1, 24, full_t, latent_h, latent_w), seed_val, rand_device, pipe.torch_dtype)[:,:,:latent_t_i]
                noise_rows = patchify_video(noise).to(device=device, dtype=pipe.torch_dtype)
                clean_part = clean_cond_rows[i * frame_rows: (i + 1) * frame_rows]
                parts.append(timestep_tensor * clean_part + (1.0 - timestep_tensor) * noise_rows)
            cond_anchor = torch.cat(parts, dim=0).contiguous()

        return {"keyframe_cond_anchor": cond_anchor, "keyframe_indices_validated": list(keyframe_indices)}


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _resolve_reference_image_shape(width: int, height: int):
    src_w, src_h = float(width), float(height)
    if src_w <= 0.0 or src_h <= 0.0:
        raise ValueError("reference image width and height must be positive")
    if src_w > 4.0 * src_h or src_h > 4.0 * src_w:
        raise ValueError(f"reference image ratio must be within 1:4 to 4:1, got {width}x{height}")
    scale = REFERENCE_IMAGE_SHORT_EDGE / min(src_w, src_h)
    return (_nearest_multiple(src_w * scale, REFERENCE_IMAGE_MULTIPLE), _nearest_multiple(src_h * scale, REFERENCE_IMAGE_MULTIPLE))


def _resolve_reference_video_shape(width: int, height: int):
    src_w, src_h = float(width), float(height)
    if src_w <= 0.0 or src_h <= 0.0:
        raise ValueError("reference video width and height must be positive")
    ratio = src_w / src_h
    if ratio >= 1.0:
        nominal_h = float(BASE_SHORT_EDGE)
        nominal_w = BASE_SHORT_EDGE * ratio
    else:
        nominal_w = float(BASE_SHORT_EDGE)
        nominal_h = BASE_SHORT_EDGE / ratio
    nominal_area = nominal_w * nominal_h
    if nominal_area > MAX_PIXELS:
        scale = float(np.sqrt(float(MAX_PIXELS) / nominal_area))
        nominal_w *= scale
        nominal_h *= scale
    return (_nearest_multiple(nominal_w, CANVAS_MULTIPLE), _nearest_multiple(nominal_h, CANVAS_MULTIPLE))


def _trim_reference_video_length(pipe: MiniMaxH3Pipeline, frame_count: int) -> int:
    frame_count = int(frame_count)
    factor, remainder = pipe.time_division_factor, pipe.time_division_remainder
    chunks = max(1, (frame_count - remainder) // factor)
    used = chunks * factor + remainder
    if used > frame_count:
        raise ValueError(f"cannot trim {frame_count} reference video frames down to a valid " f"length: at least {used} frames are required")
    return used


class MiniMaxH3Unit_ReferenceEncoder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("references", "seed", "video_latents", "num_frames"),
            output_params=("ref_blocks",),
            onload_model_names=("video_vae", "audio_vae")
        )

    def _encode_image_ref(self, pipe, img: Image.Image):
        img = img.convert("RGB")
        target_w, target_h = _resolve_reference_image_shape(*img.size)
        if (target_w, target_h) != img.size:
            img = img.resize((target_w, target_h), Image.LANCZOS)

        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        z = pipe.video_vae.encode_video(img_tensor.to(pipe.device), dtype=pipe.torch_dtype, process_image=True,)  # [1,24,1,H',W']
        rows = patchify_video(z)
        return rows, int(z.shape[-2]), int(z.shape[-1]), img

    def _encode_video_ref(self, pipe, frames, target_frame_count: int):
        frames = [f.convert("RGB") for f in frames]
        target_w, target_h = _resolve_reference_video_shape(*frames[0].size)
        if (target_w, target_h) != frames[0].size:
            frames = [f.resize((target_w, target_h), Image.LANCZOS) for f in frames]
        # Keep at most the target number of leading frames.
        if len(frames) > int(target_frame_count):
            frames = frames[: int(target_frame_count)]
        prepared_frames = frames

        used = _trim_reference_video_length(pipe, len(prepared_frames))
        frames_np = np.stack([np.asarray(f) for f in prepared_frames[:used]], axis=0)
        frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        z = pipe.video_vae.encode_video(frames_tensor.to(pipe.device), dtype=pipe.torch_dtype, process_image=False,)  # [1,24,T',H',W']
        rows = patchify_video(z)
        return rows, int(z.shape[2]), int(z.shape[3]), int(z.shape[4]), prepared_frames

    def _encode_audio_ref(self, pipe, waveform, sample_rate: int):
        latent = _encode_audio_waveform(pipe, waveform, sample_rate)  # [C, 32, T]
        return pack_audio(latent), int(latent.shape[-1])

    @staticmethod
    def _require(ref, key, kind):
        value = ref.get(key)
        if value is None:
            raise ValueError(f"reference type {kind!r} requires field {key!r}")
        return value

    def _build_block(self, pipe, ref, target_frame_count):
        kind = ref["type"]
        if kind == "image":
            rows, lh, lw, prepared = self._encode_image_ref(pipe, self._require(ref, "image", kind))
            return {"kind": kind, "visual_clean": rows, "latent_t": 1, "latent_h": lh, "latent_w": lw, "prepared_image": prepared, "ref_audio_t": 0}
        if kind in ("video", "video_audio"):
            if kind == "video" and ref.get("audio") is not None:
                raise ValueError("reference type 'video' is silent; use 'video_audio' to pass a soundtrack")
            frames = self._require(ref, "video", kind)
            if kind == "video_audio":
                audio_waveform = self._require(ref, "audio", kind)
                audio_sample_rate = int(self._require(ref, "sample_rate", kind))
            rows, lt, lh, lw, prepared = self._encode_video_ref(pipe, frames, target_frame_count)
            block = {"kind": kind, "visual_clean": rows, "latent_t": lt, "latent_h": lh, "latent_w": lw, "prepared_frames": prepared, "ref_audio_t": 0}
            if kind == "video_audio":
                audio_rows, ref_audio_t = self._encode_audio_ref(pipe, audio_waveform, audio_sample_rate)
                block["audio_clean"] = audio_rows
                block["ref_audio_t"] = ref_audio_t
            return block
        if kind == "audio":
            rows, ref_audio_t = self._encode_audio_ref(pipe, self._require(ref, "audio", kind), int(self._require(ref, "sample_rate", kind)))
            return {"kind": kind, "audio_clean": rows, "ref_audio_t": ref_audio_t}
        raise ValueError(f"unknown reference type {kind!r}; supported: image / video / audio / video_audio")

    def process(self, pipe: MiniMaxH3Pipeline, references, seed, video_latents, num_frames):
        if not references:
            return {}

        pipe.load_models_to_device(("video_vae",))
        seed_val = int(seed) if seed is not None else 42
        device = pipe.device
        target_latent_t = video_latents.shape[2]

        ref_blocks_out = [self._build_block(pipe, ref, int(num_frames)) for ref in references]
        visual_blocks = [b for b in ref_blocks_out if "visual_clean" in b]
        imgvid_cond_num_frames = len(visual_blocks)
        noise_aug = IMGVID_COND_NOISE_AUG
        for block in visual_blocks:
            clean = block.pop("visual_clean").to(device=device, dtype=pipe.torch_dtype)
            if noise_aug == 1.0:
                anchor = clean
            else:
                full_t = target_latent_t + imgvid_cond_num_frames
                noise = pipe.generate_noise((1, 24, full_t, int(block["latent_h"]), int(block["latent_w"])), seed=seed_val, rand_device="cpu", rand_torch_dtype=pipe.torch_dtype, device="cpu", torch_dtype=pipe.torch_dtype)[:,:,: int(block["latent_t"])]
                noise_rows = patchify_video(noise).to(device=device, dtype=pipe.torch_dtype)
                ts = torch.tensor(noise_aug, dtype=pipe.torch_dtype, device=device)
                anchor = ts * clean + (1.0 - ts) * noise_rows
            block["visual_rows"] = anchor

        audio_noise_aug = AUDIO_COND_NOISE_AUG
        for block in ref_blocks_out:
            if "audio_clean" not in block or int(block["ref_audio_t"]) <= 0:
                block.pop("audio_clean", None)
                continue
            clean = block.pop("audio_clean").to(device=device, dtype=pipe.torch_dtype)
            if audio_noise_aug == 1.0:
                anchor = clean
            else:
                noise = pipe.generate_noise(clean.shape, seed=seed_val + 1, rand_device="cpu", rand_torch_dtype=pipe.torch_dtype, device=device, torch_dtype=pipe.torch_dtype)
                ts = torch.tensor(audio_noise_aug, dtype=pipe.torch_dtype, device=device)
                anchor = ts * clean + (1.0 - ts) * noise
            block["audio_rows"] = anchor

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
            seperate_cfg=True,
            input_params_posi={"prompt_embeds": "prompt_embeds", "text_token_tags": "text_token_tags"},
            input_params_nega={"prompt_embeds": "prompt_embeds", "text_token_tags": "text_token_tags"},
            input_params=("video_latents", "audio_latents", "keyframe_cond_anchor", "keyframe_indices_validated", "ref_blocks"),
            output_params=("packed",)
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
        spans = torch.tensor([cls._FRAME_RESCALE * cls._FRAME_PER_TOKEN[k % cls._T_GROUP] for k in range(n)], dtype=torch.float64)
        return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

    @classmethod
    def _temporal_position_span(cls, temporal_length: int) -> float:
        spans = np.ones(int(temporal_length), dtype=np.float64) * cls._FRAME_RESCALE
        for token_index in range(cls._T_GROUP):
            spans[token_index::cls._T_GROUP] *= cls._FRAME_PER_TOKEN[token_index]
        return float(spans.sum())

    @classmethod
    def _video_t_span(cls, n: int) -> float:
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
        g[audio_sl, 2] = torch.cat([torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64), torch.full((audio_rows - audio_t,), float(w_grid[-1]), dtype=torch.float64)])

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
            "latent_w_patched": pw
        }

    @classmethod
    def _build_packed_fl2va(cls, text_len, latent_t, latent_h, latent_w, audio_t, keyframe_indices, audio_channel=2):
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
        img_pos = torch.cat([torch.arange(cond_sl.start, cond_sl.stop), torch.arange(video_sl.start, video_sl.stop)])
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
        g[audio_sl, 2] = torch.cat([torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64), torch.full((audio_rows - audio_t,), float(w_grid[-1]), dtype=torch.float64)])

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

        # First pass: per-block dims and total ref rows, to size the sequence.
        # Video-bearing blocks pack their audio rows immediately BEFORE their video
        # rows; `ref_audio_t` may be 0 (silent video), which degrades to an empty
        # audio slice. Ref: target packed_sequence.py:336-376.
        block_dims = []
        total_ref_visual_rows = 0
        total_ref_audio_rows = 0
        for b in ref_blocks:
            kind = b["kind"]
            info = {"kind": kind, "visual_rows": 0, "audio_rows": 0, "ref_audio_t": int(b.get("ref_audio_t", 0))}
            if kind in ("image", "video", "video_audio"):
                lh_r = int(b["latent_h"])
                lw_r = int(b["latent_w"])
                lt_r = int(b["latent_t"])
                frame_rows_r = (lh_r // cls._PATCH_H) * (lw_r // cls._PATCH_W)
                info["visual_rows"] = lt_r * frame_rows_r
                info["frame_rows"] = frame_rows_r
                info["latent_t"] = lt_r
                info["latent_h"] = lh_r
                info["latent_w"] = lw_r
                total_ref_visual_rows += info["visual_rows"]
                info["audio_rows"] = info["ref_audio_t"] * audio_channel
                total_ref_audio_rows += info["audio_rows"]
            elif kind == "audio":
                info["audio_rows"] = info["ref_audio_t"] * audio_channel
                total_ref_audio_rows += info["audio_rows"]
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

            elif kind in ("video", "video_audio"):
                a_rows = info["audio_rows"]
                v_rows = info["visual_rows"]
                ref_at = info["ref_audio_t"]
                lt_r = info["latent_t"]
                lh_r = info["latent_h"]
                lw_r = info["latent_w"]
                sqrt_area = float(np.sqrt(lh_r * lw_r))
                rv_h_grid = cls._axis_from_sqrt_area(lh_r, cls._PATCH_H, sqrt_area)
                rv_w_grid = cls._axis_from_sqrt_area(lw_r, cls._PATCH_W, sqrt_area)
                hh, ww = torch.meshgrid(rv_h_grid, rv_w_grid, indexing="ij")
                frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

                audio_sl = slice(cursor, cursor + a_rows)
                visual_sl = slice(audio_sl.stop, audio_sl.stop + v_rows)

                input_ids[audio_sl] = cls._AUDIO_REF_COND_ID
                a_t_grid = t_cursor + torch.arange(ref_at, dtype=torch.float64)
                g[audio_sl, 0] = a_t_grid.repeat(audio_channel)
                if ref_at:
                    # W-axis uses THIS reference video's own grid, not the target's.
                    g[audio_sl, 2] = torch.cat([torch.full((ref_at,), float(rv_w_grid[0]), dtype=torch.float64), torch.full((a_rows - ref_at,), float(rv_w_grid[-1]), dtype=torch.float64)])
                    token_tags[audio_sl] = 2
                    ref_audio_pos_parts.append(torch.arange(audio_sl.start, audio_sl.stop))

                input_ids[visual_sl] = cls._IMGVID_COND_ID
                video_g = torch.empty(lt_r, info["frame_rows"], 3, dtype=torch.float64)
                video_g[:, :, 0] = cls._video_t_grid(lt_r, t_cursor)[:, None]
                video_g[:, :, 1:] = frame[None]
                g[visual_sl] = video_g.reshape(-1, 3)
                token_tags[visual_sl] = 0
                ref_visual_pos_parts.append(torch.arange(visual_sl.start, visual_sl.stop))

                cursor = visual_sl.stop
                t_cursor += max(float(ref_at), cls._video_t_span(lt_r))

            elif kind == "audio":
                a_rows = info["audio_rows"]
                ref_at = info["ref_audio_t"]
                sl = slice(cursor, cursor + a_rows)
                input_ids[sl] = cls._AUDIO_REF_COND_ID
                a_t_grid = t_cursor + torch.arange(ref_at, dtype=torch.float64)
                g[sl, 0] = a_t_grid.repeat(audio_channel)
                if ref_at:
                    g[sl, 2] = torch.cat([torch.full((ref_at,), float(target_w_grid[0]), dtype=torch.float64), torch.full((a_rows - ref_at,), float(target_w_grid[-1]), dtype=torch.float64)])
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
        g[target_audio_sl, 2] = torch.cat([torch.full((audio_t,), float(w_grid_t[0]), dtype=torch.float64), torch.full((target_audio_rows - audio_t,), float(w_grid_t[-1]), dtype=torch.float64)])

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

    def process(self, pipe: MiniMaxH3Pipeline, prompt_embeds, video_latents, audio_latents, text_token_tags=None, keyframe_cond_anchor=None, keyframe_indices_validated=None, ref_blocks=None):
        text_len = int(prompt_embeds.shape[0])
        video_latent_t, latent_h, latent_w = (int(x) for x in video_latents.shape[2:])
        audio_latent_t = int(audio_latents.shape[-1])
        if ref_blocks is not None:
            packed = self._build_packed_ref2va(text_len, video_latent_t, latent_h, latent_w, audio_latent_t, ref_blocks)
        elif keyframe_cond_anchor is not None:
            packed = self._build_packed_fl2va(text_len, video_latent_t, latent_h, latent_w, audio_latent_t, keyframe_indices_validated)
        else:
            packed = self._build_packed_t2va(text_len, video_latent_t, latent_h, latent_w, audio_latent_t)

        dev = pipe.device
        if text_token_tags is not None:
            tags = text_token_tags.view(-1).to(torch.long)
            if int(tags.shape[0]) != text_len:
                raise ValueError(f"text_token_tags length {int(tags.shape[0])} != text_len {text_len}")
            packed["token_tags"][packed["text_pos"]] = tags.cpu()
        packed["img_pos"] = packed["img_pos"].to(dev, torch.long)
        packed["audio_pos"] = packed["audio_pos"].to(dev, torch.long)
        packed["text_pos"] = packed["text_pos"].to(dev, torch.long)
        packed["update_mask"] = packed["update_mask"].to(dev)
        if "audio_update_mask" in packed:
            packed["audio_update_mask"] = packed["audio_update_mask"].to(dev)
        if pipe.device == "mps": packed["img_position_ids"] = packed["img_position_ids"].to(torch.float32)
        packed["img_position_ids"] = packed["img_position_ids"][None].to(dev)  # [1,S,3] fp64
        packed["token_tags"] = packed["token_tags"].to(dev, torch.long)
        packed["cu_seqlens"] = packed["cu_seqlens"].to(dev, torch.int32)
        return {"packed": packed}


def model_fn_minimax_h3(
    dit,
    video_latents,
    audio_latents,
    packed,
    prompt_embeds,
    t_video,
    t_audio,
    device,
    keyframe_cond_anchor=None,
    ref_blocks=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    # Every producer already hands these over on `device` at the pipeline dtype, so
    # the packed buffers just follow the latents. index_copy_ below enforces that.
    dtype = video_latents.dtype
    video_rows = patchify_video(video_latents)
    audio_rows = pack_audio(audio_latents)

    seq_len = packed["seq_len"]
    img_pos = packed["img_pos"]
    audio_pos = packed["audio_pos"]
    text_pos = packed["text_pos"]
    cu = packed["cu_seqlens"]
    text_len = packed["text_len"]
    update_mask = packed["update_mask"]
    cond_rows_count = packed.get("cond_rows", 0)
    ref_audio_rows_count = packed.get("ref_audio_rows", 0)

    x = torch.zeros(1, seq_len, 96, dtype=dtype, device=device)
    audio_x = torch.zeros(1, seq_len, 32, dtype=dtype, device=device)

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
            ref_visual_anchor = torch.cat(visual_parts, dim=0)
        if audio_parts:
            ref_audio_anchor = torch.cat(audio_parts, dim=0)

    if ref_visual_anchor is not None and cond_rows_count > 0:
        ref_pos = img_pos[:cond_rows_count]
        target_pos = img_pos[cond_rows_count:]
        x[0].index_copy_(0, target_pos, video_rows)
        x[0].index_copy_(0, ref_pos, ref_visual_anchor)
    elif keyframe_cond_anchor is not None and cond_rows_count > 0:
        cond_pos = img_pos[:cond_rows_count]
        target_pos = img_pos[cond_rows_count:]
        x[0].index_copy_(0, target_pos, video_rows)
        x[0].index_copy_(0, cond_pos, keyframe_cond_anchor)
    else:
        x[0].index_copy_(0, img_pos, video_rows)

    if ref_audio_anchor is not None and ref_audio_rows_count > 0:
        ref_audio_pos = audio_pos[:ref_audio_rows_count]
        target_audio_pos = audio_pos[ref_audio_rows_count:]
        audio_x[0].index_copy_(0, target_audio_pos, audio_rows)
        audio_x[0].index_copy_(0, ref_audio_pos, ref_audio_anchor)
    else:
        audio_x[0].index_copy_(0, audio_pos, audio_rows)

    timesteps = torch.full((seq_len,), float(t_video), dtype=torch.float32, device=device)
    timesteps[audio_pos] = float(t_audio)
    if cond_rows_count > 0:
        cond_t = max(float(t_video), IMGVID_COND_NOISE_AUG)
        timesteps[img_pos[:cond_rows_count]] = cond_t
    if ref_audio_rows_count > 0:
        audio_ref_t = max(float(t_audio), AUDIO_COND_NOISE_AUG)
        timesteps[audio_pos[:ref_audio_rows_count]] = audio_ref_t

    unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)

    refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)
    v_video_rows, v_audio_rows = dit(
        x=x,
        audio_x=audio_x,
        img_position_ids=packed["img_position_ids"],
        unique_timesteps=unique_timesteps,
        inverse_indices=inverse_indices,
        update_mask=update_mask,
        token_tags=packed["token_tags"],
        prompt_embeds=prompt_embeds,
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

    if cond_rows_count > 0:
        v_video_rows = v_video_rows[cond_rows_count:]
    if ref_audio_rows_count > 0:
        v_audio_rows = v_audio_rows[ref_audio_rows_count:]

    v_video = unpatchify_video(v_video_rows, packed["latent_t"], packed["latent_h_patched"], packed["latent_w_patched"])
    v_audio = unpack_audio(v_audio_rows, packed["audio_channel"], packed["audio_t"])
    return -v_video, -v_audio
