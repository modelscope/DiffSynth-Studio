from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.enable_vram_management(enable_dit_fp8_computation=True)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40, enable_fp8_attention=True)
image.save("image.jpg")
