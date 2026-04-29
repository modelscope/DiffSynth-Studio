from diffsynth.pipelines.ernie_image import ErnieImagePipeline, ModelConfig
import torch

pipe = ErnieImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device='cuda',
    model_configs=[
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image-Turbo", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="tokenizer/"),
)

image = pipe(
    prompt="一只黑白相间的中华田园犬",
    negative_prompt="",
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=8,
    cfg_scale=1.0,
    sigma_shift=4.0,
)
image.save("output_turbo.jpg")
