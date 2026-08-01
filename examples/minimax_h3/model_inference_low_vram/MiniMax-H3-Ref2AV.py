import os
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

import torch
import numpy as np
import av
from PIL import Image
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

MODEL_ID = "MiniMaxAI/MiniMax-H3"

vram_config = {
    "offload_dtype": "disk", "offload_device": "disk",
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
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
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

ref_image_raw = Image.open("assets_minimax/ref2av/example1/image.png")
ref_image = ref_image_raw.resize((TARGET_W, TARGET_H), Image.LANCZOS)

ref_video_raw = load_video_frames("assets_minimax/ref2av/example1/ref1.mov", max_frames=22)
ref_video_frames = [
    np.array(Image.fromarray(f).resize((TARGET_W, TARGET_H), Image.LANCZOS))
    for f in ref_video_raw
]
prompt = open("assets_minimax/ref2av/example1/prompt.txt").read().strip()

video, audio = pipe(
    prompt=prompt,
    height=TARGET_H, width=TARGET_W, num_frames=24,
    num_inference_steps=50, seed=42,
    references=[
        {"type": "image", "data": ref_image},
        {"type": "video", "data": ref_video_frames},
    ],
)
write_video_audio(
    video=video,
    audio=audio,
    output_path="minimax_h3_ref2av_ex1_low_vram.mp4",
    fps=24,
    audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_ref2av_ex1_low_vram.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
