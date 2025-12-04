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
        ModelConfig(model_id="DiffSynth-Studio/LoRA-Encoder-FLUX.1-Dev", origin_file_pattern="model.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
lora = ModelConfig(model_id="VoidOc/flux_animal_forest1", origin_file_pattern="20.safetensors")
pipe.load_lora(pipe.dit, lora) # Use `pipe.clear_lora()` to drop the loaded LoRA.

# Empty prompt can automatically activate LoRA capabilities.
image = pipe(prompt="", seed=0, lora_encoder_inputs=lora)
image.save("image_1.jpg")

image = pipe(prompt="", seed=0)
image.save("image_1_origin.jpg")

# Prompt without trigger words can also activate LoRA capabilities.
image = pipe(prompt="a car", seed=0, lora_encoder_inputs=lora)
image.save("image_2.jpg")

image = pipe(prompt="a car", seed=0,)
image.save("image_2_origin.jpg")

# Adjust the activation intensity through the scale parameter.
image = pipe(prompt="a cat", seed=0, lora_encoder_inputs=lora, lora_encoder_scale=1.0)
image.save("image_3.jpg")

image = pipe(prompt="a cat", seed=0, lora_encoder_inputs=lora, lora_encoder_scale=0.5)
image.save("image_3_scale.jpg")
