import os
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")  # env flash_attn is a stub
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")  # load pre-downloaded local files

import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

# Normal inference still needs VRAM management (DiT ~33B + text encoder ~25.7B).
# bf16, params resident on CPU, streamed to CUDA for compute.
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
        ModelConfig(model_id="MiniMaxAI/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMaxAI/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMaxAI/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMaxAI/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
prompt = "A girl is very happy, she is speaking: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"

video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=120,
    num_inference_steps=50, seed=42,
)
write_video_audio(
    video=video,
    audio=audio,
    output_path="minimax_h3_t2va.mp4",
    fps=24,
    audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_t2va.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
