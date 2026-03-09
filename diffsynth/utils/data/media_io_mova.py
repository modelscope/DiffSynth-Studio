import os
import shutil
import subprocess
import tempfile
import wave

import imageio
import numpy as np
import torch
from tqdm import tqdm
from ..data import merge_video_audio
try:
    import imageio_ffmpeg as _imageio_ffmpeg
except ImportError:  # pragma: no cover
    _imageio_ffmpeg = None


def _write_wav_wave(audio, wav_path, sample_rate=44100):
    """
    Write int16 PCM WAV using standard library wave.
    - audio: torch.Tensor or np.ndarray, shape [samples] / [channels, samples]
      - If float, assumed range is approximately [-1, 1], will be converted to int16 PCM
    """
    if isinstance(audio, torch.Tensor):
        a = audio.detach().cpu().numpy()
    else:
        a = np.asarray(audio)

    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"audio shape needs to be [S] / [C,S], current shape is {a.shape}")

    channels, samples = int(a.shape[0]), int(a.shape[1])
    if channels > 2:
        a = a[:2, :]
        channels = 2

    if np.issubdtype(a.dtype, np.floating):
        a = np.clip(a, -1.0, 1.0)
        a = (a * 32767.0).astype(np.int16)
    elif a.dtype != np.int16:
        a = np.clip(a, -32768, 32767).astype(np.int16)

    if channels == 1:
        interleaved = a.reshape(-1)
    else:
        interleaved = a.T.reshape(-1)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(sample_rate))
        wf.writeframes(interleaved.tobytes(order="C"))


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


# Copied from https://github.com/sgl-project/sglang/blob/7106f6c8e1509cd57abeafd5d50cb1beaffbc63c/python/sglang/multimodal_gen/runtime/entrypoints/utils.py#L96
def _resolve_ffmpeg_exe() -> str:
    ffmpeg_exe = "ffmpeg"
    ffmpeg_on_path = shutil.which("ffmpeg")
    if ffmpeg_on_path:
        ffmpeg_exe = ffmpeg_on_path
    try:
        if _imageio_ffmpeg is not None:
            ffmpeg_exe = _imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    ffmpeg_ok = False
    if ffmpeg_exe:
        if os.path.isabs(ffmpeg_exe):
            ffmpeg_ok = os.path.exists(ffmpeg_exe)
        else:
            ffmpeg_ok = shutil.which(ffmpeg_exe) is not None
    if not ffmpeg_ok:
        raise RuntimeError("ffmpeg not found")
    return ffmpeg_exe

def save_video_with_audio(frames, audio_data, save_path, fps=24, sample_rate=44100, quality=9, ffmpeg_path=None):
    """
    Save video with audio.
    - frames: List[PIL.Image | np.ndarray]
    - audio: torch.Tensor or np.ndarray, shape [channels, samples] or [samples]
    - save_path: Output mp4 path
    - fps: Video frame rate
    - sample_rate: Audio sample rate (default 44100)
    Depend on ffmpeg executable program for audio/video reuse.
    """
    if ffmpeg_path is None:
        ffmpeg_path = _resolve_ffmpeg_exe()

    with tempfile.TemporaryDirectory(prefix='save_vwa_') as tmp_dir:
        tmp_audio = os.path.join(tmp_dir, 'audio.wav')
        save_video(frames, save_path, fps=fps, quality=quality)
        _write_wav_wave(audio_data, tmp_audio, sample_rate=sample_rate)
        merge_video_audio(save_path, tmp_audio)
