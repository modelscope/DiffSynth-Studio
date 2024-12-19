import torch
torch.cuda.set_per_process_memory_fraction(1.0, 0)
from diffsynth import ModelManager, HunyuanVideoPipeline, download_models, save_video


download_models(["HunyuanVideo"])
model_manager = ModelManager()

# The DiT model is loaded in bfloat16.
model_manager.load_models(
    [
        "models/HunyuanVideo/transformers/mp_rank_00_model_states.pt"
    ],
    torch_dtype=torch.bfloat16, # you can use torch_dtype=torch.float8_e4m3fn to enable quantization.
    device="cpu"
)

# The other modules are loaded in float16.
model_manager.load_models(
    [
        "models/HunyuanVideo/text_encoder/model.safetensors",
        "models/HunyuanVideo/text_encoder_2",
        "models/HunyuanVideo/vae/pytorch_model.pt",
    ],
    torch_dtype=torch.float16,
    device="cpu"
)

# We support LoRA inference. You can use the following code to load your LoRA model.
# model_manager.load_lora("models/lora/xxx.safetensors", lora_alpha=1.0)

# The computation device is "cuda".
pipe = HunyuanVideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device="cuda"
)

# Enjoy!
prompt = "CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."
video = pipe(prompt, seed=0)
save_video(video, "video_girl.mp4", fps=30, quality=6)
