from diffsynth.pipelines.ming_image import MingImagePipeline, ModelConfig
import torch

pipe = MingImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="mllm/model*.safetensors"),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="connector/model*.safetensors"),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="mlp/model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="mllm/"),
)
pipe.load_lora(pipe.dit, "models/train/Ming-Image-0.2-Design_lora/epoch-4.safetensors")
prompt = "dog,white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt=prompt, height=1024, width=1024, seed=42, num_inference_steps=12, cfg_scale=1.0)
image.save("image_Ming-Image-0.2-Design_lora.png")
