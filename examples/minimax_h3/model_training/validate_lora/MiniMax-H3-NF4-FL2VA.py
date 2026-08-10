import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from diffsynth.utils.data import VideoData
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-fl2va-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="minimax_h3/MiniMax-H3-FL2VA/*",
)
dataset_base_path = "data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA"
height, width, num_frames = 480, 832, 124

prompt = "A girl is very happy, she is speaking in english: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
pipe.load_lora(pipe.dit, "models/train/MiniMax-H3-T2VA-nf4/epoch-4.safetensors")
video, audio = pipe(
    prompt=prompt,
    height=height, width=width, num_frames=num_frames,
    num_inference_steps=50, seed=0,
)
write_video_audio(
    video=video, audio=audio, output_path="minimax_h3_t2va_lora_nf4.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_t2va_lora_nf4.mp4", "frames:", len(video), "audio:", tuple(audio.shape))

frames = VideoData(f"{dataset_base_path}/video.mp4", height=height, width=width).raw_data()
pipe.clear_lora()
pipe.load_lora(pipe.dit, "models/train/MiniMax-H3-FL2VA-nf4/epoch-4.safetensors")
video, audio = pipe(
    prompt=prompt,
    height=height, width=width, num_frames=num_frames,
    num_inference_steps=50, seed=0,
    keyframes=[frames[0], frames[num_frames - 1]],
    keyframe_indices=[0, -1],
)
write_video_audio(
    video=video, audio=audio, output_path="minimax_h3_fl2va_lora_nf4.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_fl2va_lora_nf4.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
