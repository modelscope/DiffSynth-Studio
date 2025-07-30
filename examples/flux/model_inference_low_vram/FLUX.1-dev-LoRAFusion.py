import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="DiffSynth-Studio/FLUX.1-dev-LoRAFusion", origin_file_pattern="model.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn)
    ],
)
pipe.enable_vram_management()
pipe.enable_lora_patcher()
pipe.load_lora(
    pipe.dit,
    ModelConfig(model_id="yangyufeng/fgao", origin_file_pattern="30.safetensors"),
    hotload=True
)
pipe.load_lora(
    pipe.dit,
    ModelConfig(model_id="bobooblue/LoRA-bling-mai", origin_file_pattern="10.safetensors"),
    hotload=True
)
pipe.load_lora(
    pipe.dit,
    ModelConfig(model_id="JIETANGAB/E", origin_file_pattern="17.safetensors"),
    hotload=True
)

image = pipe(prompt="This is a digital painting in a soft, ethereal style. a beautiful Asian girl Shine like a diamond. Everywhere is shining with bling bling luster.The background is a textured blue with visible brushstrokes, giving the image an impressionistic style reminiscent of Vincent van Gogh's work", seed=0)
image.save("flux.jpg")
