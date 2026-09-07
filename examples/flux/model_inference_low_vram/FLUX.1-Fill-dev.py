import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from PIL import Image
from modelscope import dataset_snapshot_download

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
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="flux1-fill-dev.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="ae.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

dataset_snapshot_download(
    dataset_id="HuanJue/example_dataset",
    local_dir="./",
    allow_file_pattern=f"FLUX.1-Fill-dev/*",
)

flux_fill_image = Image.open("FLUX.1-Fill-dev/cup.png").convert("RGB")
flux_fill_mask = Image.open("FLUX.1-Fill-dev/cup_mask.png").convert("L")
prompt = "a white paper cup"
image = pipe(prompt=prompt, flux_fill_image=flux_fill_image, flux_fill_mask=flux_fill_mask, height=1632, width=1232, seed=0, embedded_guidance=30.0, num_inference_steps=50)

image.save("image_FLUX.1-Fill-dev.jpg")
