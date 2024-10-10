from diffsynth import ModelManager, CogVideoPipeline, save_video, download_models
import torch


download_models(["CogVideoX-5B", "ExVideo-CogVideoX-LoRA-129f-v1"])
model_manager = ModelManager(torch_dtype=torch.bfloat16)
model_manager.load_models([
    "models/CogVideo/CogVideoX-5b/text_encoder",
    "models/CogVideo/CogVideoX-5b/transformer",
    "models/CogVideo/CogVideoX-5b/vae/diffusion_pytorch_model.safetensors",
])
model_manager.load_lora("models/lora/ExVideo-CogVideoX-LoRA-129f-v1.safetensors")
pipe = CogVideoPipeline.from_model_manager(model_manager)

torch.manual_seed(6)
video = pipe(
    prompt="an astronaut riding a horse on Mars.",
    height=480, width=720, num_frames=129,
    cfg_scale=7.0, num_inference_steps=100,
)
save_video(video, "video_with_lora.mp4", fps=8, quality=5)
