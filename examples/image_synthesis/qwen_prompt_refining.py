from diffsynth import ModelManager, SDXLImagePipeline, download_models, QwenPrompt
import torch


download_models(["StableDiffusionXL_v1", "QwenPrompt"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/sd_xl_base_1.0.safetensors",
    "models/QwenPrompt/qwen2-1.5b-instruct",
])
pipe = SDXLImagePipeline.from_model_manager(model_manager, prompt_refiner_classes=[QwenPrompt])

prompt = "一个漂亮的女孩"
negative_prompt = ""

for seed in range(4):
    torch.manual_seed(seed)
    image = pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        height=1024, width=1024,
        num_inference_steps=30
    )
    image.save(f"{seed}.jpg")
