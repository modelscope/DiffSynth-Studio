from diffsynth.pipelines.ernie_image import ErnieImagePipeline, ModelConfig
from diffsynth.core.device.npu_compatible_device import get_device_type
import torch

pipe = ErnieImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=get_device_type(),
    model_configs=[
        ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="pe/model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="tokenizer/"),
    pe_tokenizer_config=ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="pe/"),
)

image, revised_prompt = pipe(
    prompt="一只黑白相间的中华田园犬",
    negative_prompt="",
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=50,
    cfg_scale=4.0,
    use_pe=True,
)
image.save("output.jpg")
print(f"Revised prompt: {revised_prompt}")
