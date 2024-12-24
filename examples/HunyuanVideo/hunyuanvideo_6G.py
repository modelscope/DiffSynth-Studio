import torch
torch.cuda.set_per_process_memory_fraction(1.0, 0)
from diffsynth import ModelManager, HunyuanVideoPipeline, download_models, save_video, FlowMatchScheduler, download_customized_models


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
# Example LoRA: https://civitai.com/models/1032126/walking-animation-hunyuan-video
download_customized_models(
    model_id="AI-ModelScope/walking_animation_hunyuan_video",
    origin_file_path="kxsr_walking_anim_v1-5.safetensors",
    local_dir="models/lora"
)[0]
model_manager.load_lora("models/lora/kxsr_walking_anim_v1-5.safetensors", lora_alpha=1.0)

# The computation device is "cuda".
pipe = HunyuanVideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device="cuda"
)
# This LoRA requires shift=9.0.
pipe.scheduler = FlowMatchScheduler(shift=9.0, sigma_min=0.0, extra_one_step=True)

# Enjoy!
for clothes_up in ["white t-shirt", "black t-shirt", "orange t-shirt"]:
    for clothes_down in ["blue sports skirt", "red sports skirt", "white sports skirt"]:
        prompt = f"kxsr, full body, no crop, A 3D-rendered CG animation video featuring a Gorgeous, mature, curvaceous, fair-skinned female girl with long silver hair and blue eyes. She wears a {clothes_up} and a {clothes_down}, walking offering a sense of fluid movement and vivid animation."
        video = pipe(prompt, seed=0, height=512, width=384, num_frames=129, num_inference_steps=18, tile_size=(17, 16, 16), tile_stride=(12, 12, 12))
        save_video(video, f"video-{clothes_up}-{clothes_down}.mp4", fps=30, quality=6)
