from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
import torch


vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors", quantize=QuantizeConfig(method="bitsandbytes_nf4"), **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", quantize=QuantizeConfig(method="bitsandbytes_nf4"), **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
pipe.load_lora(pipe.dit, "./models/train/Z-Image-Turbo_quant_lora/epoch-4.safetensors")
prompt = "a dog"
image = pipe(prompt=prompt, seed=42, rand_device="cuda")
image.save("image.jpg")
