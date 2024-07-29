from diffsynth import ModelManager, SDXLImagePipeline, download_models, Translator, BeautifulPrompt
import torch


# Download models (automatically)
# `models/stable_diffusion_xl/sd_xl_base_1.0.safetensors`: [link](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors)
# `models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd/`: [link](https://huggingface.co/alibaba-pai/pai-bloom-1b1-text2prompt-sd)
# `models/translator/opus-mt-zh-en/`: [link](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)
download_models(["StableDiffusionXL_v1", "BeautifulPrompt", "opus-mt-zh-en"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/sd_xl_base_1.0.safetensors",
    "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd",
    "models/translator/opus-mt-zh-en"
])
pipe = SDXLImagePipeline.from_model_manager(model_manager, prompt_refiner_classes=[Translator, BeautifulPrompt])

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
