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


# ---------------------------------------------------------------------------
# Qwen3-VL presentation (ref: target presentation.py)
# ---------------------------------------------------------------------------
_VISION_START = "<|vision_start|>"
_VISION_END = "<|vision_end|>"
_IMAGE_PAD = "<|image_pad|>"
_VIDEO_PAD = "<|video_pad|>"
# AdaLN modality tags: text rows are 1, Qwen vision spans are 0 (VIDEO).
_PRESENTATION_TEXT_TAG = 1
_PRESENTATION_VIDEO_TAG = 0
# Ref: target reference_encoding.py:502-503
_QWEN_VIDEO_SAMPLE_FPS = 2.0
_QWEN_TEMPORAL_PATCH = 2
# Ref: target resolved_plan.py:39-42
_MINIMAX_H3_BASE_SHORT_EDGE = 768
_MINIMAX_H3_MAX_PIXELS = 768 * 1344
_MINIMAX_H3_CANVAS_MULTIPLE = 32
# Video VAE temporal chunking (config.json: vae_clip_length=17), 17n+5 grouping.
_MINIMAX_H3_VAE_CLIP_LENGTH = 17
_MINIMAX_H3_VAE_TAIL_FRAMES = 5


def _text_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _vision_block_ids(tokenizer, pad_token: str, count: int) -> list[int]:
    """Ref: target presentation.py:33-39."""
    return (
        [tokenizer.convert_tokens_to_ids(_VISION_START)]
        + [tokenizer.convert_tokens_to_ids(pad_token)] * int(count)
        + [tokenizer.convert_tokens_to_ids(_VISION_END)]
    )


class _Presentation:
    """Accumulates aligned (ids, token_tags) segments. Ref: target presentation.py:41-61."""

    def __init__(self):
        self.ids: list[int] = []
        self.tags: list[int] = []

    def text(self, token_ids: list[int]):
        self.ids += token_ids
        self.tags += [_PRESENTATION_TEXT_TAG] * len(token_ids)

    def vision(self, token_ids: list[int]):
        self.ids += token_ids
        self.tags += [_PRESENTATION_VIDEO_TAG] * len(token_ids)

    def build(self):
        return (
            torch.tensor(self.ids, dtype=torch.long),
            torch.tensor(self.tags, dtype=torch.long),
        )


def _presentation_t2va(tokenizer, prompt: str):
    """Verbatim prompt, all tags TEXT.

    Ref: target presentation.py:83-88 + text_encoding.py:139-149.
    """
    if not prompt:
        raise ValueError("prompt must be non-empty")
    presentation = _Presentation()
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def _presentation_fl2va(tokenizer, prompt: str, image_token_counts):
    """`<Picture i>: ` + image vision block per keyframe, then the prompt.

    Ref: target presentation.py:91-107 `_multi_image_presentation`.
    """
    if not image_token_counts:
        raise ValueError("image_token_counts must be non-empty")
    presentation = _Presentation()
    for index, count in enumerate(image_token_counts, start=1):
        if int(count) <= 0:
            raise ValueError("image_token_count must be positive")
        presentation.text(_text_ids(tokenizer, f"<Picture {index}>: "))
        presentation.vision(_vision_block_ids(tokenizer, _IMAGE_PAD, count))
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def _presentation_ref2va(
    tokenizer,
    prompt: str,
    condition_labels,
    image_token_counts,
    video_block_token_counts,
    video_block_timestamps,
):
    """Per condition in request order, then the verbatim prompt.

    image i -> `<Picture i>: ` + image vision block
    audio j -> `<Audio j>: ` label only (audio content never enters Qwen)
    video k -> `<Video k>: ` then per temporal block `<{t:.1f} seconds>` + video block

    Ref: target presentation.py:216-295 `minimax_h3_ref2va_video_presentation`.
    """
    if not prompt:
        raise ValueError("prompt must be non-empty")
    presentation = _Presentation()
    image_seen = 0
    video_seen = 0
    for cond_type, ordinal in condition_labels:
        if cond_type == "image":
            image_seen += 1
            if image_seen > len(image_token_counts):
                raise ValueError("image_token_count required for an image reference")
            count = int(image_token_counts[image_seen - 1])
            if count <= 0:
                raise ValueError("image_token_count required for an image reference")
            presentation.text(_text_ids(tokenizer, f"<Picture {ordinal}>: "))
            presentation.vision(_vision_block_ids(tokenizer, _IMAGE_PAD, count))
        elif cond_type == "audio":
            presentation.text(_text_ids(tokenizer, f"<Audio {ordinal}>: "))
        elif cond_type == "video":
            video_seen += 1
            if video_seen > len(video_block_token_counts):
                raise ValueError("video reference requires block token counts and timestamps")
            counts = video_block_token_counts[video_seen - 1]
            timestamps = video_block_timestamps[video_seen - 1]
            if not counts or len(counts) != len(timestamps):
                raise ValueError("video block token counts and timestamps must align")
            presentation.text(_text_ids(tokenizer, f"<Video {ordinal}>: "))
            for count, timestamp in zip(counts, timestamps):
                if int(count) <= 0:
                    raise ValueError("video block token count must be positive")
                presentation.text(_text_ids(tokenizer, f"<{timestamp:.1f} seconds>"))
                presentation.vision(_vision_block_ids(tokenizer, _VIDEO_PAD, count))
        else:
            raise ValueError(f"unsupported ref2va condition type {cond_type!r}")
    if image_seen != len(image_token_counts):
        raise ValueError("unused image_token_count entries")
    if video_seen != len(video_block_token_counts):
        raise ValueError("unused video block token count entries")
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def _sample_qwen_video_frames(frames):
    """Sample 2fps frames + per-block timestamps from a 24fps CFR frame list.

    Ref: target reference_encoding.py:527-571 `minimax_h3_sample_reference_video_frames`.
    The cursor/round/dedup recipe is kept verbatim; ratio is 24/2 = 12, so dedup
    never drops a wanted frame. Timestamps pad to the temporal patch with the
    last entry, then each block takes the mean of its merged pair.
    """
    ratio = _MINIMAX_H3_SUPPORTED_FPS / _QWEN_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while True:
        idx = int(round(cursor))
        if idx >= len(frames):
            break
        if not indices or idx > indices[-1]:
            indices.append(idx)
        cursor += ratio
    if not indices:
        raise ValueError("no frames sampled for the Qwen video presentation")
    ts = [i / _QWEN_VIDEO_SAMPLE_FPS for i in range(len(indices))]
    pad = (-len(ts)) % _QWEN_TEMPORAL_PATCH
    ts = ts + [ts[-1]] * pad
    block_timestamps = [
        (ts[i] + ts[i + _QWEN_TEMPORAL_PATCH - 1]) / 2
        for i in range(0, len(ts), _QWEN_TEMPORAL_PATCH)
    ]
    return [frames[i] for i in indices], block_timestamps


# ---------------------------------------------------------------------------
# Spatial / temporal policies (ref: target resolved_plan.py + reference_encoding.py)
# ---------------------------------------------------------------------------
def _nearest_multiple(value: float, multiple: int) -> int:
    """Ref: target resolved_plan.py:91-98, reference_encoding.py:100-101."""
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _resolve_reference_image_shape(width: int, height: int):
    """Reference images always target a 2048px short edge, upscaling when needed.

    Ratio must stay within 1:4 to 4:1; there is NO area cap here, unlike the
    target-canvas / reference-video policy.
    Ref: target reference_encoding.py:104-158 `minimax_h3_resolve_reference_image_shape`.
    """
    src_w, src_h = float(width), float(height)
    if src_w <= 0.0 or src_h <= 0.0:
        raise ValueError("reference image width and height must be positive")
    if src_w > 4.0 * src_h or src_h > 4.0 * src_w:
        raise ValueError(
            f"reference image ratio must be within 1:4 to 4:1, got {width}x{height}"
        )
    scale = _MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(src_w, src_h)
    return (
        _nearest_multiple(src_w * scale, _MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE),
        _nearest_multiple(src_h * scale, _MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE),
    )


def _resolve_reference_video_shape(width: int, height: int):
    """`adapt_shape_v1`: 768px short edge, capped at 768*1344 pixels, nearest-32.

    Reference videos follow the target-canvas policy, NOT the 2048 image policy.
    Ref: target prequeue.py:247-259 -> resolved_plan.py:107-175
    `minimax_h3_resolve_spatial_shape(base_short_edge=768)`.
    """
    src_w, src_h = float(width), float(height)
    if src_w <= 0.0 or src_h <= 0.0:
        raise ValueError("reference video width and height must be positive")
    ratio = src_w / src_h
    if ratio >= 1.0:
        nominal_h = float(_MINIMAX_H3_BASE_SHORT_EDGE)
        nominal_w = _MINIMAX_H3_BASE_SHORT_EDGE * ratio
    else:
        nominal_w = float(_MINIMAX_H3_BASE_SHORT_EDGE)
        nominal_h = _MINIMAX_H3_BASE_SHORT_EDGE / ratio
    nominal_area = nominal_w * nominal_h
    if nominal_area > _MINIMAX_H3_MAX_PIXELS:
        scale = float(np.sqrt(float(_MINIMAX_H3_MAX_PIXELS) / nominal_area))
        nominal_w *= scale
        nominal_h *= scale
    return (
        _nearest_multiple(nominal_w, _MINIMAX_H3_CANVAS_MULTIPLE),
        _nearest_multiple(nominal_h, _MINIMAX_H3_CANVAS_MULTIPLE),
    )


def _trim_reference_video_length(frame_count: int) -> int:
    """Trim DOWN to the largest 17n+5 (n>=1) that fits; never pad up.

    Equivalent to the video VAE's `get_suitable_video_length` with chunk-granularity
    `mode="trim"` (`vae_processor.py:100-171`). Fewer than 22 frames cannot be
    trimmed, matching `align_video_length`'s "Cannot trim ... not enough frames".
    """
    frame_count = int(frame_count)
    chunks = max(
        1, (frame_count - _MINIMAX_H3_VAE_TAIL_FRAMES) // _MINIMAX_H3_VAE_CLIP_LENGTH
    )
    used = chunks * _MINIMAX_H3_VAE_CLIP_LENGTH + _MINIMAX_H3_VAE_TAIL_FRAMES
    if used > frame_count:
        raise ValueError(
            f"cannot trim {frame_count} reference video frames down to a valid "
            f"length: at least {used} frames are required"
        )
    return used


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
        self.processor = None
        self.in_iteration_models = ("dit",)
        # PromptEmbedder runs AFTER the visual encoders so the Qwen presentation
        # consumes the exact same prepared images the VAE saw
        # (ref: target canvas.py:8-11, text_encoding.py:170-176).
        self.units = [
            MiniMaxH3Unit_ShapeChecker(),
            MiniMaxH3Unit_NoiseInitializer(),
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
        tokenizer_config: ModelConfig = None,
        processor_config: ModelConfig = None,
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
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import AutoProcessor
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
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

        Input contract: `video` frame lists must ALREADY be 24fps CFR — the
        pipeline never resamples frame rate. `sample_rate` is mandatory and is the
        only rate the pipeline normalizes (one resample to 32kHz at the audio VAE
        boundary). Use `video_audio` when a reference video carries its soundtrack;
        it is fail-closed and raises if `audio` is missing.
        """
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
            output_params=("height", "width", "num_frames",
                           "video_latent_t", "audio_latent_t", "latent_h", "latent_w"),
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
        # Freeze the aligned frame count back onto num_frames so every later unit
        # consumes one shape. Ref: target prequeue.py:280-289 `batch.num_frames = work_frames`.
        num_frames = self._align_frame_count(num_frames)
        # Audio latent length follows the ALIGNED duration.
        # Ref: target resolved_plan.py:202-211 + time_request.py:29-31.
        duration_seconds = float(num_frames) / float(_MINIMAX_H3_SUPPORTED_FPS)
        return {
            "height": height, "width": width, "num_frames": num_frames,
            "video_latent_t": self._video_latent_t(num_frames),
            "audio_latent_t": int(round(duration_seconds * 40.0)),
            "latent_h": height // 16, "latent_w": width // 16,
        }


class MiniMaxH3Unit_PromptEmbedder(PipelineUnit):
    """Encode the Qwen3-VL presentation (text + optional vision blocks).

    t2va  : verbatim prompt only.
    fl2va : `<Picture i>: ` + image vision block per keyframe, then the prompt.
    ref2va: per reference in request order `<Picture i>: ` / `<Audio j>: ` /
            `<Video k>: ` + timestamped video blocks, then the prompt.

    Ref: target text_encoding.py:130-149 (t2va), :155-218 (fl2va), :220-408 (ref2va).
    Also emits `text_token_tags` so the packed layout can tag Qwen vision spans as
    VIDEO(0) instead of TEXT(1) (ref: target denoising.py:386-390).
    """

    def __init__(self):
        super().__init__(
            input_params=("prompt", "keyframe_prepared_images", "ref_blocks"),
            output_params=("prompt_embeds", "text_token_tags"),
            onload_model_names=("text_encoder",),
        )

    @staticmethod
    def _image_token_counts(processor, images):
        """Ref: target text_encoding.py:308-323."""
        vision = processor.image_processor(images=images, return_tensors="pt")
        grid = vision["image_grid_thw"]
        if int(grid.shape[0]) != len(images):
            raise ValueError(f"expected {len(images)} image grids, got {list(grid.shape)}")
        merge = int(processor.image_processor.merge_size) ** 2
        counts = [int(grid[i].prod().item()) // merge for i in range(len(images))]
        return vision["pixel_values"], grid, counts

    @staticmethod
    def _video_token_counts(processor, videos, timestamps_per_video):
        """Ref: target text_encoding.py:349-379."""
        vout = processor.video_processor(
            videos=videos, do_sample_frames=False, return_tensors="pt",
        )
        grid = vout["video_grid_thw"]
        if int(grid.shape[0]) != len(videos):
            raise ValueError(f"expected {len(videos)} video grids, got {list(grid.shape)}")
        merge = int(processor.image_processor.merge_size) ** 2
        block_counts, block_timestamps = [], []
        for index, timestamps in enumerate(timestamps_per_video):
            n_blocks = int(grid[index, 0])
            per_block = int(grid[index, 1]) * int(grid[index, 2]) // merge
            if len(timestamps) != n_blocks:
                raise ValueError(
                    f"video block count mismatch: processor {n_blocks} vs "
                    f"timestamps {len(timestamps)} for video {index}"
                )
            block_counts.append([per_block] * n_blocks)
            block_timestamps.append([float(t) for t in timestamps])
        return vout["pixel_values_videos"], grid, block_counts, block_timestamps

    def process(self, pipe: MiniMaxH3Pipeline, prompt,
                keyframe_prepared_images=None, ref_blocks=None):
        pipe.load_models_to_device(("text_encoder",))
        pixel_values = image_grid_thw = None
        pixel_values_videos = video_grid_thw = None

        if ref_blocks:
            if pipe.processor is None:
                raise ValueError("ref2va requires processor_config in from_pretrained")
            # Condition labels are per-type 1-based; a soundtrack-bearing video
            # emits its audio label BEFORE its video label.
            # Ref: target text_encoding.py:257-294.
            counters = {"image": 0, "audio": 0, "video": 0}
            condition_labels = []
            images, videos, timestamps_per_video = [], [], []
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
                    sampled, timestamps = _sample_qwen_video_frames(block["prepared_frames"])
                    videos.append(np.stack([np.asarray(f) for f in sampled]))
                    timestamps_per_video.append(timestamps)
                else:
                    raise ValueError(f"unknown reference kind: {kind}")

            image_counts = []
            if images:
                pixel_values, image_grid_thw, image_counts = self._image_token_counts(
                    pipe.processor, images
                )
            video_counts, video_timestamps = [], []
            if videos:
                (pixel_values_videos, video_grid_thw, video_counts,
                 video_timestamps) = self._video_token_counts(
                    pipe.processor, videos, timestamps_per_video
                )
            input_ids, text_token_tags = _presentation_ref2va(
                pipe.tokenizer, prompt, condition_labels,
                image_counts, video_counts, video_timestamps,
            )
        elif keyframe_prepared_images:
            if pipe.processor is None:
                raise ValueError("fl2va requires processor_config in from_pretrained")
            pixel_values, image_grid_thw, image_counts = self._image_token_counts(
                pipe.processor, keyframe_prepared_images
            )
            input_ids, text_token_tags = _presentation_fl2va(
                pipe.tokenizer, prompt, image_counts
            )
        else:
            input_ids, text_token_tags = _presentation_t2va(pipe.tokenizer, prompt)

        ids = input_ids[None].to(pipe.device)
        kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if pixel_values is not None or pixel_values_videos is not None:
            # get_rope_index routes grids by token type: image_pad -> 1, video_pad -> 2.
            # Mislabelling a video pad as 1 makes it consume an image grid entry.
            # Ref: target minimax_h3_qwen3vl.py:198-214.
            mm_types = torch.zeros_like(ids, dtype=torch.int32)
            mm_types[ids == pipe.text_encoder.image_token_id] = 1
            mm_types[ids == pipe.text_encoder.video_token_id] = 2
            kwargs["mm_token_type_ids"] = mm_types
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values.to(pipe.device, torch.bfloat16)
            kwargs["image_grid_thw"] = image_grid_thw.to(pipe.device, torch.long)
        if pixel_values_videos is not None:
            kwargs["pixel_values_videos"] = pixel_values_videos.to(pipe.device, torch.bfloat16)
            kwargs["video_grid_thw"] = video_grid_thw.to(pipe.device, torch.long)

        hidden = pipe.text_encoder(**kwargs)
        return {
            "prompt_embeds": hidden[0].to(pipe.device, torch.bfloat16),
            "text_token_tags": text_token_tags.to(pipe.device, torch.long),
        }


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
    unit is a no-op (t2va mode). Outputs keyframe_cond_anchor (patchified rows
    with noise augmentation applied once, used as fixed anchor each denoise step)
    plus keyframe_prepared_images — the SAME prepared canvas images the Qwen
    presentation must consume (ref: target canvas.py:8-11)."""

    def __init__(self):
        super().__init__(
            input_params=("keyframes", "keyframe_indices", "latent_h", "latent_w",
                          "video_latent_t", "seed", "imgvid_cond_noise_aug"),
            output_params=("keyframe_cond_anchor", "keyframe_indices_validated",
                           "keyframe_prepared_images"),
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
        prepared_images = []
        target_w, target_h = latent_w * 16, latent_h * 16
        for img in keyframes:
            img = img.convert("RGB")
            # Identity when the image already IS the canvas (ref: target canvas.py:92-94).
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.LANCZOS)
            prepared_images.append(img)
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
            "keyframe_prepared_images": prepared_images,
        }


class MiniMaxH3Unit_ReferenceEncoder(PipelineUnit):
    """Encode Ref2AV reference conditions. No-op when `references` is None.

    `references` is a list of dicts in request order, fields named per modality:

        {"type": "image",       "image": PIL.Image}
        {"type": "video",       "video": list[PIL.Image]}                 # silent
        {"type": "audio",       "audio": Tensor[C, L], "sample_rate": int}
        {"type": "video_audio", "video": list[PIL.Image],
                                "audio": Tensor[C, L], "sample_rate": int}

    `video` frame lists must already be 24fps CFR (the pipeline never resamples
    frame rate); `sample_rate` is mandatory and the only rate the pipeline
    normalizes (resampled once to 32kHz at the audio VAE boundary).

    `video_audio` is the explicit "this video carries its soundtrack" contract and
    is fail-closed: a missing `audio` raises instead of silently degrading.
    Ref: target task_profiles.py:193-209, audio_encoding.py:114-137.

    Output `ref_blocks`: per-reference dicts carrying the noise-augmented anchor
    rows, geometry for PackedSequenceBuilder / model_fn, and the prepared media
    the Qwen presentation must reuse.
    """

    def __init__(self):
        super().__init__(
            input_params=("references", "seed", "imgvid_cond_noise_aug",
                          "audio_cond_noise_aug", "video_latent_t", "num_frames"),
            output_params=("ref_blocks",),
            onload_model_names=("video_vae", "audio_vae"),
        )

    def _encode_image_ref(self, pipe, img: Image.Image):
        """Resolve to a 2048px short edge, then run the keyframe encode recipe.

        Ref: target reference_encoding.py:104-183 (shape + resize)
             + keyframe_encoding.py:35-75 (image / keyframe share one encode recipe).
        """
        img = img.convert("RGB")
        target_w, target_h = _resolve_reference_image_shape(*img.size)
        if (target_w, target_h) != img.size:
            img = img.resize((target_w, target_h), Image.LANCZOS)

        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = pipe.video_vae.processor.transform_tensor(img_tensor)
        vae_dtype = next(pipe.video_vae.parameters()).dtype
        img_tensor = img_tensor.to(device=pipe.device, dtype=vae_dtype)
        with torch.random.fork_rng(devices=[pipe.device] if str(pipe.device) != "cpu" else []):
            torch.manual_seed(_MINIMAX_H3_KEYFRAME_ENCODE_SEED)
            z = pipe.video_vae.encode_base(img_tensor, process_image=True)  # [1,24,1,H',W']
        mean = torch.tensor(_VIDEO_LATENTS_MEAN, device=z.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(_VIDEO_LATENTS_STD, device=z.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        rows = patchify_video((z.float() - mean) / std)
        return rows, int(z.shape[-2]), int(z.shape[-1]), img

    def _encode_video_ref(self, pipe, frames, target_frame_count: int):
        """Resize by `adapt_shape_v1`, cap at the target frame count, then encode.

        Ref: target prequeue.py:247-259 (spatial policy),
             reference_encoding.py:414-425 (cap to target_frame_count),
             reference_encoding.py:459-499 (full-frame encode + normalize + patchify).

        Two frame lists come out of this on purpose: the Qwen presentation samples
        from the capped list, while the VAE consumes the same list trimmed DOWN to
        17n+5 by `get_suitable_video_length` (ref: vae_processor.py:163-171).
        """
        frames = [f.convert("RGB") for f in frames]
        target_w, target_h = _resolve_reference_video_shape(*frames[0].size)
        if (target_w, target_h) != frames[0].size:
            frames = [f.resize((target_w, target_h), Image.LANCZOS) for f in frames]
        # Keep at most the target number of leading frames.
        if len(frames) > int(target_frame_count):
            frames = frames[: int(target_frame_count)]
        prepared_frames = frames

        used = _trim_reference_video_length(len(prepared_frames))
        frames_np = np.stack([np.asarray(f) for f in prepared_frames[:used]], axis=0)
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
        rows = patchify_video((z - mean) / std)
        return rows, int(z.shape[2]), int(z.shape[3]), int(z.shape[4]), prepared_frames

    def _encode_audio_ref(self, pipe, waveform, sample_rate: int):
        """Resample to 32kHz, force stereo, then run the deterministic mean encode.

        Ref: target reference_encoding.py:278-336 `minimax_h3_encode_reference_audio_rows`.
        """
        import torchaudio  # local import; needed only for resample

        pipe.load_models_to_device(("audio_vae",))
        model = pipe.audio_vae
        device = next(model.parameters()).device

        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)  # [1,C,L] -> [C,L]
        if waveform.dim() != 2:
            raise ValueError(f"expected audio waveform [C, L], got {list(waveform.shape)}")
        waveform = waveform.float()

        if int(sample_rate) != _MINIMAX_H3_AUDIO_SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(
                int(sample_rate), _MINIMAX_H3_AUDIO_SAMPLE_RATE
            )(waveform)

        if waveform.shape[0] < _MINIMAX_H3_AUDIO_CHANNELS:
            repeats = (_MINIMAX_H3_AUDIO_CHANNELS + waveform.shape[0] - 1) // waveform.shape[0]
            waveform = waveform.repeat(repeats, 1)
        # Match the VAE's parameter dtype: VRAM management may hold weights in bf16
        # while the waveform is fp32. (Target upcasts the VAE to fp32 instead; under
        # VRAM management we cannot mutate the wrapped weights, so we cast the input.)
        vae_dtype = next(model.parameters()).dtype
        waveform = waveform[:_MINIMAX_H3_AUDIO_CHANNELS].to(device=device, dtype=vae_dtype)

        with _AudioVAEDeterminismContext():
            # Batched: preprocess([2, 1, L]) -> encoder -> optional pre_block -> mean_proj
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
            latent = latent.transpose(1, 2).contiguous()  # -> [2, T, 32]

        # Normalize per target reference_encoding.py:326-329 (view [1,1,32])
        mean = torch.tensor(_AUDIO_LATENTS_MEAN, dtype=torch.float32).view(1, 1, latent_channels)
        std = torch.tensor(_AUDIO_LATENTS_STD, dtype=torch.float32).view(1, 1, latent_channels)
        rows = ((latent - mean) / std).reshape(-1, latent_channels).to(torch.float32).contiguous()
        return rows.to(device), int(latent.shape[1])

    @staticmethod
    def _require(ref, key, kind):
        value = ref.get(key)
        if value is None:
            raise ValueError(f"reference type {kind!r} requires field {key!r}")
        return value

    def _build_block(self, pipe, ref, target_frame_count):
        kind = ref["type"]
        if kind == "image":
            rows, lh, lw, prepared = self._encode_image_ref(
                pipe, self._require(ref, "image", kind))
            return {"kind": kind, "visual_clean": rows, "latent_t": 1,
                    "latent_h": lh, "latent_w": lw, "prepared_image": prepared,
                    "ref_audio_t": 0}
        if kind in ("video", "video_audio"):
            if kind == "video" and ref.get("audio") is not None:
                raise ValueError(
                    "reference type 'video' is silent; use 'video_audio' to pass a soundtrack"
                )
            frames = self._require(ref, "video", kind)
            if kind == "video_audio":
                # Fail closed before spending a video encode.
                audio_waveform = self._require(ref, "audio", kind)
                audio_sample_rate = int(self._require(ref, "sample_rate", kind))
            rows, lt, lh, lw, prepared = self._encode_video_ref(
                pipe, frames, target_frame_count)
            block = {"kind": kind, "visual_clean": rows, "latent_t": lt,
                     "latent_h": lh, "latent_w": lw, "prepared_frames": prepared,
                     "ref_audio_t": 0}
            if kind == "video_audio":
                audio_rows, ref_audio_t = self._encode_audio_ref(
                    pipe, audio_waveform, audio_sample_rate)
                block["audio_clean"] = audio_rows
                block["ref_audio_t"] = ref_audio_t
            return block
        if kind == "audio":
            rows, ref_audio_t = self._encode_audio_ref(
                pipe, self._require(ref, "audio", kind),
                int(self._require(ref, "sample_rate", kind)))
            return {"kind": kind, "audio_clean": rows, "ref_audio_t": ref_audio_t}
        raise ValueError(
            f"unknown reference type {kind!r}; supported: image / video / audio / video_audio"
        )

    def process(self, pipe: MiniMaxH3Pipeline, references, seed, imgvid_cond_noise_aug,
                audio_cond_noise_aug, video_latent_t, num_frames):
        if not references:
            return {}

        pipe.load_models_to_device(("video_vae",))
        seed_val = int(seed) if seed is not None else 42
        device = pipe.device
        target_latent_t = int(video_latent_t)

        # First pass: encode every reference, keeping clean rows + geometry.
        ref_blocks_out = [
            self._build_block(pipe, ref, int(num_frames)) for ref in references
        ]

        # Second pass: condition noise augmentation.
        # Ref: target condition_noise.py:24-119 (visual) and :122-191 (audio).
        visual_blocks = [b for b in ref_blocks_out if "visual_clean" in b]
        imgvid_cond_num_frames = len(visual_blocks)
        noise_aug = float(imgvid_cond_noise_aug)
        for block in visual_blocks:
            clean = block.pop("visual_clean").to(device=device, dtype=torch.float32)
            if noise_aug == 1.0:
                anchor = clean
            else:
                full_t = target_latent_t + imgvid_cond_num_frames
                generator = torch.Generator(device="cpu").manual_seed(seed_val)
                noise = torch.randn(
                    1, 24, full_t, int(block["latent_h"]), int(block["latent_w"]),
                    generator=generator, dtype=torch.float32, device="cpu",
                )[:, :, : int(block["latent_t"])]
                noise_rows = patchify_video(noise).to(device=device, dtype=torch.float32)
                ts = torch.tensor(noise_aug, dtype=torch.float32, device=device)
                anchor = ts * clean + (1.0 - ts) * noise_rows
            block["visual_rows"] = anchor

        # Only conditions with ref_audio_t > 0 take a slot in the audio RNG stream.
        # Ref: target denoising.py:592-614 `_condition_audio_lengths`.
        audio_noise_aug = float(audio_cond_noise_aug)
        for block in ref_blocks_out:
            if "audio_clean" not in block or int(block["ref_audio_t"]) <= 0:
                block.pop("audio_clean", None)
                continue
            clean = block.pop("audio_clean").to(device=device, dtype=torch.float32)
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
                          "text_token_tags", "keyframe_cond_anchor",
                          "keyframe_indices_validated", "ref_blocks"),
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

        # First pass: per-block dims and total ref rows, to size the sequence.
        # Video-bearing blocks pack their audio rows immediately BEFORE their video
        # rows; `ref_audio_t` may be 0 (silent video), which degrades to an empty
        # audio slice. Ref: target packed_sequence.py:336-376.
        block_dims = []
        total_ref_visual_rows = 0
        total_ref_audio_rows = 0
        for b in ref_blocks:
            kind = b["kind"]
            info = {"kind": kind, "visual_rows": 0, "audio_rows": 0,
                    "ref_audio_t": int(b.get("ref_audio_t", 0))}
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
                # Audio rows come FIRST, then the video rows; both share this
                # block's temporal origin, and the cursor advances by the longer
                # of the two spans. Ref: target packed_sequence.py:404-410, 474-514.
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
                    g[audio_sl, 2] = torch.cat([
                        torch.full((ref_at,), float(rv_w_grid[0]), dtype=torch.float64),
                        torch.full((a_rows - ref_at,), float(rv_w_grid[-1]), dtype=torch.float64),
                    ])
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
                # Standalone audio: W-axis uses the TARGET grid.
                # Ref: target packed_sequence.py:453-473.
                a_rows = info["audio_rows"]
                ref_at = info["ref_audio_t"]
                sl = slice(cursor, cursor + a_rows)
                input_ids[sl] = cls._AUDIO_REF_COND_ID
                a_t_grid = t_cursor + torch.arange(ref_at, dtype=torch.float64)
                g[sl, 0] = a_t_grid.repeat(audio_channel)
                if ref_at:
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
                text_token_tags=None, keyframe_cond_anchor=None,
                keyframe_indices_validated=None, ref_blocks=None):
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
        # The presentation's own tags override the text segment, so Qwen vision
        # spans select the VIDEO AdaLN branch instead of TEXT.
        # Ref: target denoising.py:386-390.
        if text_token_tags is not None:
            tags = text_token_tags.view(-1).to(torch.long)
            if int(tags.shape[0]) != text_len:
                raise ValueError(
                    f"text_token_tags length {int(tags.shape[0])} != text_len {text_len}"
                )
            packed["token_tags"][packed["text_pos"]] = tags.cpu()
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
