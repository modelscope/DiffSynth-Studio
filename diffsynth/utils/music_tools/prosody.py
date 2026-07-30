import numpy as np
import librosa
import torch
from scipy.signal import butter, filtfilt
from einops import repeat


def extract_smooth_envelope(y, sr, cutoff_hz=30.0):
    """
    提取平滑的幅度包络。
    保留音节时序（20-30Hz），同时消除高频毛刺，防止电子波形产生“咔哒”点击声。
    """
    abs_y = np.abs(y)
    nyq = 0.5 * sr
    
    # 第一次低通：提取宏观音节包络
    b1, a1 = butter(4, cutoff_hz / nyq, btype='low')
    env = filtfilt(b1, a1, abs_y)
    
    # 第二次低通：平滑包络边缘，让波形过渡更自然
    b2, a2 = butter(2, 80.0 / nyq, btype='low')
    env = filtfilt(b2, a2, env)
    
    return env / (np.max(env) + 1e-8)

def fill_nan_f0(arr):
    """处理基频提取中的 NaN 值（静音或无音高区域），使用前向和后向填充"""
    mask = np.isnan(arr)
    if not np.any(mask): return arr
    
    # 前向填充
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    out = arr[idx]
    
    # 后向填充 (处理开头的 NaN)
    mask2 = np.isnan(out)
    if np.any(mask2):
        idx2 = np.where(~mask2, np.arange(mask2.shape[0]), mask2.shape[0]-1)
        np.minimum.accumulate(idx2[::-1], out=idx2[::-1])
        out = out[idx2]
    return out

# ================= 旋律跟踪正弦波 =================
def extract_prosody(audio_tensor, sr=48000):
    audio_tensor = audio_tensor.squeeze()
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=0)
    y = audio_tensor.cpu().numpy()

    envelope = extract_smooth_envelope(y, sr)

    hop_length = 512

    f0, _, _ = librosa.pyin(y, fmin=65, fmax=1000, sr=sr, hop_length=hop_length)
    f0 = fill_nan_f0(f0)

    t_frames = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    t_samples = np.arange(len(y)) / sr
    f0_samples = np.interp(t_samples, t_frames, f0)

    phase = 2 * np.pi * np.cumsum(f0_samples) / sr
    carrier = np.sin(phase)

    out = carrier * envelope
    out = out * (np.max(np.abs(y)) / (np.max(np.abs(out)) + 1e-8))

    out = torch.from_numpy(out.astype(np.float32))
    out = repeat(out, "l -> n l", n=2)
    return out
