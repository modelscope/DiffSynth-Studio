import torch
from modelscope import dataset_snapshot_download
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data import VideoData
from diffsynth.utils.data.audio_video import write_video_audio

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
        ModelConfig(model_id="DiffSynth-Studio/Diffutoon-2", origin_file_pattern="line-art_module.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/Diffutoon-2", origin_file_pattern="color_module.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
# The two ControlNets are trained in separate runs, so their LoRAs are loaded one after another.
pipe.load_lora(pipe.dit, "models/train/Diffutoon-2-Line-Art_lora/epoch-4.safetensors")
pipe.load_lora(pipe.dit, "models/train/Diffutoon-2-Color_lora/epoch-4.safetensors")

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="minimax_h3/MiniMax-H3-Fun-Controlnet-Union/*",
)
dataset_base_path = "data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union"
height, width, num_frames = 480, 832, 124

# The sample dataset provides a single control video, it is fed to both ControlNets.
# The prompt is the first row of the dataset metadata.
prompt = "A T-Rex riding a bicycle, with explosions in the background."
control_video = VideoData(f"{dataset_base_path}/control_video.mp4", height=height, width=width).raw_data()[:num_frames]

video, audio = pipe(
    prompt=prompt,
    height=height, width=width, num_frames=num_frames,
    num_inference_steps=40, seed=43,
    control_video=[control_video, control_video], control_scale=[0.4, 0.4],
)
write_video_audio(
    video=video, audio=audio, output_path="diffutoon2_lora.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved diffutoon2_lora.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
