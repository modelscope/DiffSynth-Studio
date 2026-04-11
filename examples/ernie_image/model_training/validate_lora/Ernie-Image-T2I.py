import torch
from diffsynth.pipelines.ernie_image import ErnieImagePipeline, ModelConfig
from diffsynth.core.loader.file import load_state_dict

pipe = ErnieImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
)

# Load LoRA state dict manually to avoid load_lora path resolution issues
lora_state_dict = load_state_dict("./models/train/Ernie-Image-T2I_lora/epoch-4.safetensors", torch_dtype=torch.bfloat16, device="cuda")
pipe.load_lora(pipe.dit, state_dict=lora_state_dict, alpha=1.0)

image = pipe(
    prompt="a professional photo of a cute dog",
    seed=0,
    num_inference_steps=50,
    cfg_scale=4.0,
)
image.save("image_lora.jpg")
print("LoRA validation image saved to image_lora.jpg")
