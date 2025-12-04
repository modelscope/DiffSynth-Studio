import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig


vram_config = {
    # Enable lora hotloading
    "offload_dtype": torch.bfloat16,
    "offload_device": "cuda",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev", origin_file_pattern="model.safetensors"),
    ],
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
