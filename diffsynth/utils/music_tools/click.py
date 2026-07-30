import torch

def generate_click(bpm, sample_rate=48000, duration=None, length=None, click_duration=0.01, click_freq=1000, bias=0):
    num_samples = int(duration * sample_rate) if duration is not None else length
    if duration is None: duration = length / sample_rate
    click_len = int(click_duration * sample_rate)
    t = torch.arange(click_len, dtype=torch.float64) / sample_rate
    click = 0.5 * torch.sin(2 * torch.pi * click_freq * t) * torch.exp(-t * 100)

    output = torch.zeros(num_samples, dtype=torch.float64)
    for idx in (torch.arange(0, duration, 60.0 / bpm) * sample_rate).long():
        idx = int(idx + bias * 60.0 / bpm * sample_rate)
        end = min(idx + click_len, num_samples)
        output[idx:end] += click[:end - idx]

    output /= output.abs().max()
    return output.float().unsqueeze(0).repeat(2, 1)
