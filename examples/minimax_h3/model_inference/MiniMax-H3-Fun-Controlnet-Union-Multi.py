import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/root/models"
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
        ModelConfig(model_id="PAI/MiniMax-H3-Fun-Controlnet-Union", origin_file_pattern="MiniMax-H3-Fun-Controlnet-Union.safetensors", **vram_config),
        ModelConfig(path="/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39-sigma20/step-1600.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)

pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))

dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Fun-Multi-Controlnet/*")

# One control video per ControlNet, in the order they are loaded in `model_configs`.
lineart_video = VideoData("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Multi-Controlnet/video_lineart.mp4", height=768, width=1344)
lineart_video = [lineart_video[i] for i in range(124)]
blur_video = VideoData("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Multi-Controlnet/video_blur.mp4", height=768, width=1344)
blur_video = [blur_video[i] for i in range(124)]
prompt = "Generate a Japanese anime-style cartoon video with clean line work and flat cel-shaded colors, following the outlines of the line-art video and the colors, lighting and motion of the blurred video.\ndetailed_description: A cinematic outdoor scene bathed in warm, golden hour sunlight, featuring a man in a vibrant pink suit standing in a lush green field with sheep."
# The hints of all ControlNets are summed, so each scale is usually lower than when used alone.
video, audio = pipe(
    prompt=prompt,
    height=768, width=1344, num_frames=124,
    num_inference_steps=4, seed=43,
    control_video=[lineart_video, blur_video], control_scale=[0.5, 0.5],
)
write_video_audio(
    video=video, audio=audio, output_path="minimax_h3_fun_controlnet_union_multi.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
