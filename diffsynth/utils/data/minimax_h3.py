import os

from ...core.data.operators import DataProcessingOperator, ImageCropAndResize, LoadImage, LoadVideo
from ...models.minimax_constant import (
    MINIMAX_H3_AUDIO_SAMPLE_RATE,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_SUPPORTED_FPS,
    MINIMAX_H3_VAE_CLIP_LENGTH,
    MINIMAX_H3_VAE_TAIL_FRAMES,
)
from .audio import read_audio

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "bmp")
VIDEO_EXTENSIONS = ("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm")
AUDIO_EXTENSIONS = ("mp3", "wav", "flac", "m4a", "aac", "ogg")


class MiniMaxH3ReferenceLoader(DataProcessingOperator):
    """Turns a dataset `references` column into the reference blocks `MiniMaxH3Pipeline` accepts.

    Each entry is either an explicit spec or a bare path whose type is inferred from its
    extension. Both a single entry and a list of entries are accepted:

        "0.png"
        {"type": "image",       "image": "0.png"}
        {"type": "video",       "video": "clip.mp4"}
        {"type": "audio",       "audio": "voice.mp3"}
        {"type": "video_audio", "video": "clip.mp4", "audio": "clip.mp4"}

    Reference images are handed over at native resolution because the pipeline rescales them to
    its own reference short edge; reference videos are cropped to the training canvas and sampled
    at the pipeline's fixed frame rate, mirroring what the inference examples do.
    """

    def __init__(
        self,
        base_path="",
        height=None,
        width=None,
        max_pixels=None,
        num_frames=124,
        frame_rate=MINIMAX_H3_SUPPORTED_FPS,
        audio_sample_rate=MINIMAX_H3_AUDIO_SAMPLE_RATE,
    ):
        self.base_path = base_path
        self.frame_rate = frame_rate
        self.audio_sample_rate = audio_sample_rate
        self.max_duration = num_frames / frame_rate
        self.image_loader = LoadImage()
        self.video_loader = LoadVideo(
            num_frames=num_frames,
            time_division_factor=MINIMAX_H3_VAE_CLIP_LENGTH,
            time_division_remainder=MINIMAX_H3_VAE_TAIL_FRAMES,
            frame_processor=ImageCropAndResize(
                height=height,
                width=width,
                max_pixels=max_pixels,
                height_division_factor=MINIMAX_H3_CANVAS_MULTIPLE,
                width_division_factor=MINIMAX_H3_CANVAS_MULTIPLE,
            ),
            frame_rate=frame_rate,
            fix_frame_rate=True,
        )

    def absolute_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.base_path, path)

    @staticmethod
    def infer_type(path):
        extension = path.rsplit(".", 1)[-1].lower()
        if extension in IMAGE_EXTENSIONS:
            return "image"
        if extension in VIDEO_EXTENSIONS:
            return "video"
        if extension in AUDIO_EXTENSIONS:
            return "audio"
        raise ValueError(f"cannot infer a reference type from {path!r}; pass an explicit spec instead")

    def load_audio(self, path, duration):
        waveform, sample_rate = read_audio(
            self.absolute_path(path), duration=duration,
            resample=True, resample_rate=self.audio_sample_rate,
        )
        return waveform, sample_rate

    def load_block(self, spec):
        if isinstance(spec, str):
            spec = {"type": self.infer_type(spec), self.infer_type(spec): spec}
        kind = spec.get("type")
        if kind is None:
            raise ValueError(f"reference spec {spec!r} is missing 'type'")

        block = {"type": kind}
        if kind == "image":
            block["image"] = self.image_loader(self.absolute_path(spec["image"]))
        elif kind == "video":
            block["video"] = self.video_loader(self.absolute_path(spec["video"]))
        elif kind == "audio":
            block["audio"], block["sample_rate"] = self.load_audio(spec["audio"], self.max_duration)
        elif kind == "video_audio":
            block["video"] = self.video_loader(self.absolute_path(spec["video"]))
            # Keep the soundtrack the same length as the frames that survived sampling.
            block["audio"], block["sample_rate"] = self.load_audio(
                spec["audio"], len(block["video"]) / self.frame_rate,
            )
        else:
            raise ValueError(f"unknown reference type {kind!r}; supported: image / video / audio / video_audio")
        return block

    def __call__(self, data):
        specs = data if isinstance(data, list) else [data]
        return [self.load_block(spec) for spec in specs]
