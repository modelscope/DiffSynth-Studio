from diffsynth import ModelManager, SD3ImagePipeline, download_models, load_state_dict
import torch


# Download models (automatically)
# `models/stable_diffusion_3/sd3_medium_incl_clips.safetensors`: [link](https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips.safetensors)
# `models/textual_inversion/verybadimagenegative_v1.3.pt`: [link](https://civitai.com/api/download/models/25820?type=Model&format=PickleTensor&size=full&fp=fp16)
download_models(["StableDiffusion3_without_T5", "TextualInversion_VeryBadImageNegative_v1.3"])
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_textual_inversions("models/textual_inversion")
model_manager.load_models(["models/stable_diffusion_3/sd3_medium_incl_clips.safetensors"])
pipe = SD3ImagePipeline.from_model_manager(model_manager)


for seed in range(4):
    torch.manual_seed(seed)
    image = pipe(
        prompt="a girl, highly detailed, absurd res, perfect image",
        negative_prompt="verybadimagenegative_v1.3",
        cfg_scale=4.5,
        num_inference_steps=50, width=1024, height=1024,
    )
    image.save(f"image_with_textual_inversion_{seed}.jpg")

    torch.manual_seed(seed)
    image = pipe(
        prompt="a girl, highly detailed, absurd res, perfect image",
        negative_prompt="",
        cfg_scale=4.5,
        num_inference_steps=50, width=1024, height=1024,
    )
    image.save(f"image_without_textual_inversion_{seed}.jpg")
