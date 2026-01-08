from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from PIL import Image
import torch


pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Omni-Base", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Omni-Base", origin_file_pattern="siglip/model.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)

pipe.load_lora(pipe.dit, "./models/train/Z-Image-Omni-Base_lora/epoch-4.safetensors")
prompt = "a dog"
image = pipe(prompt=prompt, seed=42, rand_device="cuda", num_inference_steps=40, cfg_scale=4)
image.save("image.jpg")

# Edit
# pipe.load_lora(pipe.dit, "./models/train/Z-Image-Omni-Base_lora_edit/epoch-4.safetensors")
# prompt = "Change the color of the dress in Figure 1 to the color shown in Figure 2."
# images = [
#     Image.open("data/example_image_dataset/edit/image1.jpg").resize((1024, 1024)),
#     Image.open("data/example_image_dataset/edit/image_color.jpg").resize((1024, 1024)),
# ]
# image = pipe(prompt=prompt, seed=42, rand_device="cuda", num_inference_steps=40, cfg_scale=4, edit_image=images)
# image.save("image.jpg")
