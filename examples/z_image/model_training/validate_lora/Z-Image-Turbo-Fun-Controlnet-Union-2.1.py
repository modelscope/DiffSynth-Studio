from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig, ControlNetInput
from diffsynth import load_state_dict
from PIL import Image
import torch


pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1", origin_file_pattern="Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.dit, "./models/train/Z-Image-Turbo-Fun-Controlnet-Union-2.1_lora/epoch-4.safetensors")

controlnet_image = Image.open("data/example_image_dataset/canny/image_1.jpg").resize((1024, 1024))
prompt = "a dog"
image = pipe(prompt=prompt, seed=0, height=1024, width=1024, controlnet_inputs=[ControlNetInput(image=controlnet_image, scale=0.7)])
image.save("image_control.jpg")
