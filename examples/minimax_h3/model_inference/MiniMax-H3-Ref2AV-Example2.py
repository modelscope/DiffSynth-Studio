import os
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

import torch
import numpy as np
import av
import torchaudio
from PIL import Image
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from diffsynth.utils.data.audio import read_audio

MODEL_ID = "MiniMaxAI/MiniMax-H3"

vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="Ref2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="Ref2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/video_vae/source/model.safetensors"),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="Ref2VA/audio_vae/model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id=MODEL_ID, origin_file_pattern="Ref2VA/tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)


def load_video_frames(path, max_frames=None):
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(np.array(frame.to_image()))
        if max_frames is not None and len(frames) >= max_frames:
            break
    container.close()
    return frames


TARGET_H, TARGET_W = 768, 1344

# Example 2: video + audio reference
# - video.mp4: reference video (motion/timing reference)
# - voice.mp3: reference audio (voice/timbre reference)
# Prompt: "角色说话：Follow the wind, live free.Leave worries behind, enjoy the moment，音色参考音频1"

ref_video_raw = load_video_frames("assets_minimax/ref2av/example2/video.mp4", max_frames=124)
ref_video_frames = [
    np.array(Image.fromarray(f).resize((TARGET_W, TARGET_H), Image.LANCZOS))
    for f in ref_video_raw
]

# Load reference audio waveform (torchaudio + torchcodec handles mp3 → [C, L] fp32).
# ref_audio_waveform, ref_audio_sr = torchaudio.load("assets_minimax/ref2av/example2/voice.mp3")
ref_audio_waveform, ref_audio_sr = read_audio("assets_minimax/ref2av/example2/voice.mp3", duration=5, resample=True, resample_rate=pipe.audio_vae.sample_rate)
# Pipeline internally resamples to 32kHz and ensures stereo per target library.

prompt = open("assets_minimax/ref2av/example2/prompt.txt").read().strip()

video, audio = pipe(
    prompt=prompt,
    height=TARGET_H, width=TARGET_W, num_frames=124,
    num_inference_steps=50, seed=42,
    references=[
        {"type": "video", "data": ref_video_frames},
        {"type": "audio", "data": ref_audio_waveform, "sample_rate": ref_audio_sr},
    ],
)
write_video_audio(
    video=video,
    audio=audio,
    output_path="minimax_h3_ref2av_ex2.mp4",
    fps=24,
    audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_ref2av_ex2.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
