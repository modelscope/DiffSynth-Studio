import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
pipe.load_lora(pipe.dit, "models/train/Qwen-Image-Edit_lora/epoch-4.safetensors")

prompt = "将裙子改为粉色"
image = Image.open("data/example_image_dataset/edit/image1.jpg").resize((1024, 1024))
image = pipe(prompt, edit_image=image, seed=0, num_inference_steps=40, height=1024, width=1024)
image.save(f"image.jpg")
