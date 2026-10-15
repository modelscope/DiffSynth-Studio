from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig, ControlNetInput
from PIL import Image
import torch

pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Qwen-Image-2.1-Fun-Controlnet-Union", origin_file_pattern="Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
)
pipe.load_lora(pipe.dit, "./models/train/Qwen-Image-2.1-Fun-Controlnet-Union_lora/epoch-4.safetensors")

controlnet_image = Image.open("data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1-Fun-Controlnet-Union/canny/image_1.jpg")
prompt = "a dog"
image = pipe(
    prompt=prompt, seed=0, height=1024, width=1024,
    controlnet_inputs=[ControlNetInput(image=controlnet_image, scale=0.7)],
    num_inference_steps=40,
)
image.save("image_control.png")
