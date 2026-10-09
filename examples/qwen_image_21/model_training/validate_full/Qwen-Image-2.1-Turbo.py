from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
import torch


pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path="models/train/Qwen-Image-2.1-Turbo_full/epoch-1.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
)
sigmas = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
prompt = "dog,white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt, seed=0, height=1024, width=1024, sigmas=sigmas)
image.save("image.png")
