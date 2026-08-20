import torch
import torchaudio


def convert_to_mono(audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to mono by averaging channels.
    Supports [C, T] or [B, C, T]. Output shape: [1, T] or [B, 1, T].
    """
    return audio_tensor.mean(dim=-2, keepdim=True)


def convert_to_stereo(audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to stereo.
    Supports [C, T] or [B, C, T]. Duplicate mono, keep stereo.
    """
    if audio_tensor.size(-2) == 1:
        return audio_tensor.repeat(1, 2, 1) if audio_tensor.dim() == 3 else audio_tensor.repeat(2, 1)
    return audio_tensor


def resample_waveform(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    """Resample waveform to target sample rate if needed."""
    if source_rate == target_rate:
        return waveform
    resampled = torchaudio.functional.resample(waveform, source_rate, target_rate)
    return resampled.to(dtype=waveform.dtype)


def read_audio_with_torchcodec(
    path: str,
    start_time: float = 0,
    duration: float | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Read audio from file natively using torchcodec, with optional start time and duration.
    
    Args:
        path (str): The file path to the audio file.
        start_time (float, optional): The start time in seconds to read from. Defaults to 0.
        duration (float | None, optional): The duration in seconds to read. If None, reads until the end. Defaults to None.
        
    Returns:
        tuple[torch.Tensor, int]: A tuple containing the audio tensor and the sample rate.
            The audio tensor shape is [C, T] where C is the number of channels and T is the number of audio frames.
    """
    from torchcodec.decoders import AudioDecoder
    decoder = AudioDecoder(path)
    stop_seconds = None if duration is None else start_time + duration
    waveform = decoder.get_samples_played_in_range(start_seconds=start_time, stop_seconds=stop_seconds).data
    return waveform, decoder.metadata.sample_rate


def read_audio_with_pyav(
    path: str,
    start_time: float = 0,
    duration: float | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Read audio from file using PyAV, with optional start time and duration.

    Args:
        path (str): The file path to the audio file.
        start_time (float, optional): The start time in seconds to read from. Defaults to 0.
        duration (float | None, optional): The duration in seconds to read. If None, reads until the end. Defaults to None.

    Returns:
        tuple[torch.Tensor, int]: A tuple containing the audio tensor and the sample rate.
            The audio tensor shape is [C, T] where C is the number of channels and T is the number of audio frames.
    """
    import av
    import numpy as np
    container = av.open(path)
    try:
        stream = container.streams.audio[0]
        sample_rate = stream.rate
        if start_time > 0:
            container.seek(int(start_time / float(stream.time_base)), stream=stream)
        target_samples = None if duration is None else int(duration * sample_rate)
        frames, num_samples = [], 0
        for frame in container.decode(audio=0):
            # `to_ndarray` follows the sample format: integer formats such as s16 and
            # s32 come back as integers, so they are scaled to [-1, 1] to match the
            # normalized float waveform the other backends return.
            data = frame.to_ndarray()  # [C, T]
            if np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.float32) / float(-np.iinfo(data.dtype).min)
            if target_samples is not None:
                if num_samples >= target_samples:
                    break
                data = data[:, : target_samples - num_samples]
            frames.append(data)
            num_samples += data.shape[1]
    finally:
        container.close()
    if len(frames) == 0:
        raise RuntimeError(f"No audio frames found in {path}.")
    return torch.from_numpy(np.concatenate(frames, axis=1)).float(), sample_rate


def read_audio(
    path: str,
    start_time: float = 0,
    duration: float | None = None,
    resample: bool = False,
    resample_rate: int = 48000,
    backend: str = "torchcodec",
) -> tuple[torch.Tensor, int]:
    """
    Read audio from file, with optional start time, duration, and resampling.
    
    Args:
        path (str): The file path to the audio file.
        start_time (float, optional): The start time in seconds to read from. Defaults to 0.
        duration (float | None, optional): The duration in seconds to read. If None, reads until the end. Defaults to None.
        resample (bool, optional): Whether to resample the audio to a different sample rate. Defaults to False.
        resample_rate (int, optional): The target sample rate for resampling if resample is True. Defaults to 48000.
        backend (str, optional): The audio backend to use for reading, either "torchcodec" or "pyav". Defaults to "torchcodec".

    Returns:
        tuple[torch.Tensor, int]: A tuple containing the audio tensor and the sample rate.
            The audio tensor shape is [C, T] where C is the number of channels and T is the number of audio frames.
    """
    if backend == "torchcodec":
        waveform, sample_rate = read_audio_with_torchcodec(path, start_time, duration)
    elif backend == "pyav":
        waveform, sample_rate = read_audio_with_pyav(path, start_time, duration)
    else:
        raise ValueError(f"Unsupported audio backend: {backend}")

    if resample:
        waveform = resample_waveform(waveform, sample_rate, resample_rate)
        sample_rate = resample_rate

    return waveform, sample_rate


def save_audio(waveform: torch.Tensor, sample_rate: int, save_path: str, backend: str = "torchcodec"):
    """
    Save audio tensor to file.
    
    Args:
        waveform (torch.Tensor): The audio tensor to save. Shape can be [C, T] or [B, C, T].
        sample_rate (int): The sample rate of the audio.
        save_path (str): The file path to save the audio to.
        backend (str, optional): The audio backend to use for saving. Defaults to "torchcodec".
    """
    if waveform.dim() == 3:
        waveform = waveform[0]
    waveform = waveform.cpu()

    if backend == "torchcodec":
        from torchcodec.encoders import AudioEncoder
        encoder = AudioEncoder(waveform, sample_rate=sample_rate)
        encoder.to_file(dest=save_path)
    else:
        raise ValueError(f"Unsupported audio backend: {backend}")
