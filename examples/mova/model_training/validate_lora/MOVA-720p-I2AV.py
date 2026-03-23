import torch
from PIL import Image
from diffsynth.utils.data.audio_video import write_video_audio
from diffsynth.pipelines.mova_audio_video import MovaAudioVideoPipeline, ModelConfig
from diffsynth.utils.data import VideoData


vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MovaAudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="openmoss/MOVA-720p", origin_file_pattern="video_dit/diffusion_pytorch_model-*.safetensors", **vram_config),
        ModelConfig(model_id="openmoss/MOVA-720p", origin_file_pattern="video_dit_2/diffusion_pytorch_model-*.safetensors", **vram_config),
        ModelConfig(model_id="openmoss/MOVA-720p", origin_file_pattern="audio_dit/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="openmoss/MOVA-720p", origin_file_pattern="dual_tower_bridge/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="openmoss/MOVA-720p", origin_file_pattern="audio_vae/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/Wan-Series-Converted-Safetensors", origin_file_pattern="Wan2.1_VAE.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/Wan-Series-Converted-Safetensors", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="openmoss/MOVA-720p", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.video_dit, "models/train/MOVA-720p-I2AV_high_noise_lora/epoch-4.safetensors")
negative_prompt = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
)
prompt = "A beautiful sunset over the ocean."
height, width, num_frames = 720, 1280, 121
frame_rate = 24
input_image = VideoData("data/example_video_dataset/ltx2/video.mp4", height=height, width=width)[0]
# Image-to-video
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    input_image=input_image,
    num_inference_steps=50,
    seed=0,
    tiled=True,
    frame_rate=frame_rate,
)
write_video_audio(video, audio, "MOVA-720p.mp4", fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
