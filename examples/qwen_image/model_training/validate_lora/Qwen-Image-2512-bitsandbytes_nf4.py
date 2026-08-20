from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
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
exclude_modules = ["img_mod.1", "txt_mod.1", "norm_out.linear", "img_in", "txt_in", "proj_out", "timestep_embedder.linear_1", "timestep_embedder.linear_2"]
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-2512", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", quantize=QuantizeConfig(method="bitsandbytes_nf4", exclude_modules=exclude_modules), **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
pipe.load_lora(pipe.dit, "models/train/Qwen-Image-2512_quant_lora/epoch-4.safetensors")
prompt = "a dog"
image = pipe(prompt, seed=0)
image.save("image.jpg")
