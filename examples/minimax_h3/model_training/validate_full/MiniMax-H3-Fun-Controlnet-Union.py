import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data import VideoData
from diffsynth.utils.data.audio_video import write_video_audio
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
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
        ModelConfig(path="models/train/MiniMax-H3-Fun-Controlnet-Union-full/epoch-1.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Fun-Controlnet-Union/*")
dataset_base_path = "data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union"
height, width, num_frames = 480, 832, 124
control_video = VideoData(f"{dataset_base_path}/control_video.mp4", height=height, width=width)
control_video = [control_video[i] for i in range(num_frames)]
prompt = "A young woman with long wavy red hair gently turns her head to the right, offering a soft greeting to someone off-camera. She wears an olive-green knit poncho, and the soft diffused light of dusk falls across her face. Distant building silhouettes stand against a cloudy sky."

video, audio = pipe(
    prompt=prompt,
    height=height, width=width, num_frames=num_frames,
    num_inference_steps=40, seed=43,
    control_video=control_video, control_scale=1.0,
)
write_video_audio(
    video=video, audio=audio, output_path="minimax_h3_fun_controlnet_union_full.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_fun_controlnet_union_full.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
