import av
from fractions import Fraction
import torch
from PIL import Image
from tqdm import tqdm
from . import VideoData
from .audio import convert_to_stereo, read_audio


def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the *target*
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)


def _write_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, samples: torch.Tensor, audio_sample_rate: int
) -> None:
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    samples = convert_to_stereo(samples)
    assert samples.ndim == 2 and samples.shape[0] == 2, "audio samples must be [C, S] or [S], C must be 1 or 2"
    samples = samples.T
    # Convert to int16 packed for ingestion; resampler converts to encoder fmt.
    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in)


def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac")
    supported_sample_rates = audio_stream.codec_context.codec.audio_rates
    if supported_sample_rates:
        best_rate = min(supported_sample_rates, key=lambda x: abs(x - audio_sample_rate))
        if best_rate != audio_sample_rate:
            print(f"Using closest supported audio sample rate: {best_rate}")
    else:
        best_rate = audio_sample_rate
    audio_stream.codec_context.sample_rate = best_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, best_rate)
    return audio_stream


def write_video_audio(
    video: list[Image.Image],
    audio: torch.Tensor | None,
    output_path: str,
    fps: int = 24,
    audio_sample_rate: int | None = None,
) -> None:
    """
    Writes a sequence of images and an audio tensor to a video file.

    This function utilizes PyAV (or a similar multimedia library) to encode a list of PIL images into a video stream
    and multiplex a PyTorch tensor as the audio stream into the output container.

    Args:
        video (list[Image.Image]): A list of PIL Image objects representing the video frames. 
            The length of this list determines the total duration of the video based on the FPS.
        audio (torch.Tensor | None): The audio data as a PyTorch tensor.
            The shape is typically (channels, samples). If no audio is required, pass None.
            channels can be 1 or 2. 1 for mono, 2 for stereo.
        output_path (str): The file path (including extension) where the output video will be saved.
        fps (int, optional): The frame rate (frames per second) for the video. Defaults to 24.
        audio_sample_rate (int | None, optional): The sample rate (e.g., 44100, 48000) for the audio.
            If the audio tensor is provided and this is None, the function attempts to infer the rate 
            based on the audio tensor's length and the video duration.
    Raises:
        ValueError: If an audio tensor is provided but the sample rate cannot be determined.
    """
    duration = len(video) / fps
    if audio_sample_rate is None:
        audio_sample_rate = int(audio.shape[-1] / duration)

    width, height = video[0].size
    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")
        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    for frame in tqdm(video, total=len(video)):
        frame = av.VideoFrame.from_image(frame)
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate)

    container.close()

def read_video_audio(
    path: str,
    height: int,
    width: int,
    num_frames: int | None = None,
    duration: float | None = None,
    fps: int = 24,
    audio_sample_rate: int | None = None,
) -> tuple[list[Image.Image], torch.Tensor, int]:
    """
    Read one file's video frames and its own soundtrack as a time-aligned pair.

    Frames are resampled to `fps` by nearest-source-frame lookup, and the audio is then
    clipped to the span those frames actually cover. That second step is why the two are
    read together: a source shorter than requested yields fewer frames than asked for, and
    taking the audio duration from the decoded count rather than from the request keeps the
    streams aligned.

    Args:
        path (str): A video file carrying an audio stream.
        height (int): Target frame height; frames are center-cropped and resized to it.
        width (int): Target frame width.
        num_frames (int | None): How many frames to keep, at `fps`. Give this or `duration`.
        duration (float | None): Target length in seconds, converted to a frame count via `fps`.
        fps (int, optional): Target frame rate. Defaults to 24.
        audio_sample_rate (int | None, optional): Resample the audio to this rate. Left at
            None the file's own rate is kept and the returned rate reports it.

    Returns:
        tuple[list[Image.Image], torch.Tensor, int]: The frames, the waveform as [C, T], and
            the waveform's sample rate.
    """
    if (num_frames is None) == (duration is None):
        raise ValueError("pass exactly one of `num_frames` or `duration`")
    if num_frames is None:
        num_frames = max(1, int(round(duration * fps)))

    video = VideoData(path, height=height, width=width)
    frames = video.raw_data()
    source_fps = float(video.data.reader.get_meta_data()["fps"])
    sampled = []
    for k in range(num_frames):
        index = int(round(k * source_fps / fps))
        if index >= len(frames):
            break
        sampled.append(frames[index])

    waveform, sample_rate = read_audio(path, duration=len(sampled) / fps, resample=audio_sample_rate is not None, resample_rate=audio_sample_rate)
    return sampled, waveform, sample_rate
