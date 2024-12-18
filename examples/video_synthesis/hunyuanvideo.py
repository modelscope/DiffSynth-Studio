from diffsynth import ModelManager, HunyuanVideoPipeline, download_models, save_video
import torch


# Download models (automatically)
download_models(["HunyuanVideo"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/HunyuanVideo/vae/pytorch_model.pt",
    "t2i_models/HunyuanVideo/text_encoder/model.safetensors",
    "t2i_models/HunyuanVideo/text_encoder_2",
])
pipe = HunyuanVideoPipeline.from_model_manager(model_manager)
prompt = 'A cat walks on the grass, realistic style.'
frames = pipe(prompt)
save_video(frames, 'test_video.mp4', fps=8, quality=5)
