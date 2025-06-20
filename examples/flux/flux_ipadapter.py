import torch
from PIL import Image
from diffsynth import save_video, VideoData, download_models
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download

#TODO: repalce the local path with model_id
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="InstantX/FLUX.1-dev-IP-Adapter", origin_file_pattern="ip-adapter.bin"),
        ModelConfig(path="models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/image_encoder")
    ],
)

seed = 42
origin_prompt = "a rabbit in a garden, colorful flowers"
image = pipe(prompt=origin_prompt, height=1280, width=960, seed=seed)
image.save("style image.jpg")

torch.manual_seed(seed)
image = pipe(prompt="A piggy", height=1280, width=960, seed=seed,
    ipadapter_images=[image], ipadapter_scale=0.7)
image.save("A piggy.jpg")
