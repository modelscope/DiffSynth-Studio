import torch
from diffsynth import ModelManager, FluxImagePipeline, download_customized_models

prompt = "a beautiful Asian girl."

# Generate an image using FLUX.1-dev
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev"])
pipe = FluxImagePipeline.from_model_manager(model_manager)

image = pipe(prompt=prompt, seed=0)
image.save("image.jpg")

# Download ArtAug LoRA
lora_path = download_customized_models(
    model_id="DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1",
    origin_file_path="merged_lora.safetensors",
    local_dir="models/lora",
    downloading_priority=["ModelScope", "HuggingFace"]
)[0]
model_manager.load_lora(lora_path, lora_alpha=1.0)

# Generate an image using FLUX.1-dev + ArtAug
image = pipe(prompt=prompt, seed=0)
image.save("image_artaug.jpg")
