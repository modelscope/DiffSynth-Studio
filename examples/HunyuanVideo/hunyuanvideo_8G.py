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
    torch_dtype=torch.bfloat16,
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
model_manager.load_lora("models/lora/Rem_hunyuan_video_v3.safetensors", lora_alpha=1.0)

# The computation device is "cuda".
pipe = HunyuanVideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device="cuda"
)

# Enjoy!
prompt = "a woman with blue hair wearing a white and black dress, sitting on a bed with a white wall in the background. she is wearing a re:zero starting life in another world rem cosplay costume, complete with a black and white dress, black gloves, and a black bow tie."
video = pipe(prompt, seed=0, height=512, width=512, tile_size=(17, 16, 16), tile_stride=(12, 12, 12))
save_video(video, "video.mp4", fps=30, quality=5)
