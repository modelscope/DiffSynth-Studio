import os

from ...core.data.operators import DataProcessingOperator, ImageCropAndResize, LoadImage, LoadVideo
from .audio import read_audio


class MiniMaxH3ReferenceLoader(DataProcessingOperator):
    """Turns a dataset `references` column into the reference blocks `MiniMaxH3Pipeline` accepts.

    Each entry is an explicit spec. Both a single entry and a list of entries are accepted:

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
        frame_rate=24,
        audio_sample_rate=32000,
    ):
        self.base_path = base_path
        self.frame_rate = frame_rate
        self.audio_sample_rate = audio_sample_rate
        self.max_duration = num_frames / frame_rate
        self.time_division_factor = 17
        self.time_division_remainder = 5
        self.image_loader = LoadImage()
        self.video_loader = LoadVideo(
            num_frames=num_frames,
            time_division_factor=self.time_division_factor,
            time_division_remainder=self.time_division_remainder,
            frame_processor=ImageCropAndResize(
                height=height,
                width=width,
                max_pixels=max_pixels,
                height_division_factor=32,
                width_division_factor=32,
            ),
            frame_rate=frame_rate,
            fix_frame_rate=True,
        )

    def absolute_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.base_path, path)

    def load_video(self, path):
        """Loads `path` and trims the frames down to the VAE's 17n+5 grouping.

        `LoadVideo` only snaps to that grouping when the source is shorter than `num_frames`,
        so a request such as 120 comes back untrimmed. The pipeline trims the picture again
        before encoding, so trimming here keeps the soundtrack -- cut against the frame count
        this returns -- on the same clock as the frames that survive.
        """
        frames = self.video_loader(self.absolute_path(path))
        factor, remainder = self.time_division_factor, self.time_division_remainder
        used = max(1, (len(frames) - remainder) // factor) * factor + remainder
        if used > len(frames):
            raise ValueError(f"{path!r} yields {len(frames)} frames, fewer than the {used} a reference video needs")
        return frames[:used]

    def load_audio(self, path, duration):
        waveform, sample_rate = read_audio(
            self.absolute_path(path), duration=duration,
            resample=True, resample_rate=self.audio_sample_rate,
        )
        return waveform, sample_rate

    def load_block(self, spec):
        if not isinstance(spec, dict):
            raise ValueError(f"reference spec must be a dict such as {{'type': 'image', 'image': '0.png'}}, got {spec!r}")
        kind = spec.get("type")
        if kind is None:
            raise ValueError(f"reference spec {spec!r} is missing 'type'")

        block = {"type": kind}
        if kind == "image":
            block["image"] = self.image_loader(self.absolute_path(spec["image"]))
        elif kind == "video":
            block["video"] = self.load_video(spec["video"])
        elif kind == "audio":
            block["audio"], block["sample_rate"] = self.load_audio(spec["audio"], self.max_duration)
        elif kind == "video_audio":
            block["video"] = self.load_video(spec["video"])
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
