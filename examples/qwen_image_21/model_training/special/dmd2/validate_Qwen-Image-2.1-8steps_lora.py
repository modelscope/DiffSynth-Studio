import torch
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig

lora_checkpoint = "workspace/your_training_run/step-1000.safetensors"

pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
)

pipe.load_lora(pipe.dit, lora_checkpoint)

prompt = "dog,white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt, height=1024, width=1024, seed=0, num_inference_steps=8, cfg_scale=1.0)
image.save("image_qwen_image21_dmd2_lora_8steps.png")
