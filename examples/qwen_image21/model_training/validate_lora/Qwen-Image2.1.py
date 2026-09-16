from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
import torch


model_id = "Qwen/Qwen-Image2.1"
pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id=model_id, origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id=model_id, origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id=model_id, origin_file_pattern="vae/diffusion_pytorch_model*.safetensors"),
    ],
    processor_config=ModelConfig(model_id=model_id, origin_file_pattern="processor/"),
)
pipe.load_lora(pipe.dit, "models/train/Qwen-Image2.1_lora/epoch-4.safetensors")
prompt = "dog,white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt, seed=0, num_inference_steps=50, height=1024, width=1024)
image.convert("RGB").save("image.jpg")
