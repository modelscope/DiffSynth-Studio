import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
state_dict = load_state_dict("models/train/Qwen-Image-Edit-2509_full/epoch-1.safetensors")
pipe.dit.load_state_dict(state_dict)

prompt = "Change the color of the dress in Figure 1 to the color shown in Figure 2."
images = [
    Image.open("data/example_image_dataset/edit/image1.jpg").resize((1024, 1024)),
    Image.open("data/example_image_dataset/edit/image_color.jpg").resize((1024, 1024)),
]
image = pipe(prompt, edit_image=images, seed=123, num_inference_steps=40, height=1024, width=1024)
image.save("image.jpg")
