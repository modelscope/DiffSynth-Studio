from diffsynth.pipelines.ernie_image import ErnieImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = ErnieImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device='cuda',
    model_configs=[
        ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="pe/model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="tokenizer/"),
    pe_tokenizer_config=ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="pe/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
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
