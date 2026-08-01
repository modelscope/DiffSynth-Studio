import os
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

import torch
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
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/video_vae/source/model.safetensors"),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/audio_vae/model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/tokenizer/"),
    processor_config=ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

first_frame = Image.open("assets_minimax/fl2av/first.png")
last_frame = Image.open("assets_minimax/fl2av/last.png")
prompt = open("assets_minimax/fl2av/prompt.txt").read().strip()

video, audio = pipe(
    prompt=prompt,
    height=1344, width=768, num_frames=24,
    num_inference_steps=50, seed=42,
    keyframes=[first_frame, last_frame],
    keyframe_indices=[0, -1],
)
write_video_audio(
    video=video,
    audio=audio,
    output_path="minimax_h3_fl2av_low_vram.mp4",
    fps=24,
    audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_fl2av_low_vram.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
