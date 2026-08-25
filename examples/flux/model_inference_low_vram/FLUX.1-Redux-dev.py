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
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Redux-dev", origin_file_pattern="image_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Redux-dev", origin_file_pattern="image_embedder/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
dataset_snapshot_download(
    dataset_id="HuanJue/example_dataset",
    local_dir="./",
    allow_file_pattern=f"FLUX.1-Redux-dev/*",
)

flux_redux_image = Image.open("FLUX.1-Redux-dev/robot.png").convert("RGB")

image = pipe(flux_redux_image=flux_redux_image, embedded_guidance=2.5, num_inference_steps=50)

image.save("image_FLUX.1-Redux-dev.jpg")
