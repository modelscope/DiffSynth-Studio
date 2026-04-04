import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig


vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev", origin_file_pattern="model.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
pipe.enable_lora_merger()

pipe.load_lora(
    pipe.dit,
    ModelConfig(model_id="cancel13/cxsk", origin_file_pattern="30.safetensors"),
)
pipe.load_lora(
    pipe.dit,
    ModelConfig(model_id="DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1", origin_file_pattern="merged_lora.safetensors"),
)
image = pipe(prompt="a cat", seed=0)
image.save("image_fused.jpg")
