import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig


offload_vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
prequant_vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Comfy-Org/MiniMax-H3", origin_file_pattern="text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", **offload_vram_config),
        ModelConfig(model_id="Comfy-Org/MiniMax-H3-Image", origin_file_pattern="diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors", **prequant_vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **offload_vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
pipe.load_lora(pipe.dit, "./models/train/MiniMax-H3-Image-Int8-Pruned-Comfy-split-lora/epoch-4.safetensors")

prompt = "a dog"
image = pipe(prompt=prompt, height=1024, width=1024, seed=0)
image.save("MiniMax-H3-Image-Int8-Pruned-Comfy_lora.jpg")
