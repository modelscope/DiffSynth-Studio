from diffsynth import ModelManager, FluxImagePipeline, download_models, QwenPrompt
import torch


download_models(["FLUX.1-dev", "QwenPrompt"])

model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors",
    "models/QwenPrompt/qwen2-1.5b-instruct",
])
pipe = FluxImagePipeline.from_model_manager(model_manager, prompt_refiner_classes=[QwenPrompt])

prompt = "鹰"
negative_prompt = ""

for seed in range(4):
    torch.manual_seed(seed)
    image = pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        height=1024, width=1024,
        num_inference_steps=30
    )
    image.save(f"{seed}.jpg")
