from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.models.qwen_image_dit import RMSNorm
from diffsynth.vram_management.layers import enable_vram_management, AutoWrappedLinear, AutoWrappedModule
import torch


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)

enable_vram_management(
    pipe.dit,
    module_map = {
        RMSNorm: AutoWrappedModule,
    },
    module_config = dict(
        offload_dtype=torch.bfloat16,
        offload_device="cuda",
        onload_dtype=torch.bfloat16,
        onload_device="cuda",
        computation_dtype=torch.bfloat16,
        computation_device="cuda",
    ),
    vram_limit=None,
)
enable_vram_management(
    pipe.dit,
    module_map = {
        torch.nn.Linear: AutoWrappedLinear,
    },
    module_config = dict(
        offload_dtype=torch.float8_e4m3fn,
        offload_device="cuda",
        onload_dtype=torch.float8_e4m3fn,
        onload_device="cuda",
        computation_dtype=torch.float8_e4m3fn,
        computation_device="cuda",
    ),
    vram_limit=None,
)

prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40, enable_fp8_attention=True)
image.save("image.jpg")
