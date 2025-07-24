import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig

        
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev", origin_file_pattern="model.safetensors"),
    ],
)
pipe.enable_lora_magic()

pipe.load_lora(
    pipe.dit,
    ModelConfig(model_id="cancel13/cxsk", origin_file_pattern="30.safetensors"),
    hotload=True,
)
pipe.load_lora(
    pipe.dit,
    ModelConfig(model_id="DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1", origin_file_pattern="merged_lora.safetensors"),
    hotload=True,
)
image = pipe(prompt="a cat", seed=0)
image.save("image_fused.jpg")
