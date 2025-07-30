import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig, ControlNetInput
from modelscope import dataset_snapshot_download
from modelscope import snapshot_download
from PIL import Image
import numpy as np


snapshot_download(
    "ByteDance/InfiniteYou",
    allow_file_pattern="supports/insightface/models/antelopev2/*",
    local_dir="models/ByteDance/InfiniteYou",
)
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="ByteDance/InfiniteYou", origin_file_pattern="infu_flux_v1.0/aes_stage2/image_proj_model.bin", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="ByteDance/InfiniteYou", origin_file_pattern="infu_flux_v1.0/aes_stage2/InfuseNetModel/*.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
    ],
)
pipe.enable_vram_management()

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=f"data/examples/infiniteyou/*",
)

height, width = 1024, 1024
controlnet_image = Image.fromarray(np.zeros([height, width, 3]).astype(np.uint8))
controlnet_inputs = [ControlNetInput(image=controlnet_image, scale=1.0, processor_id="None")]

prompt = "A man, portrait, cinematic"
id_image = "data/examples/infiniteyou/man.jpg"
id_image = Image.open(id_image).convert('RGB')
image = pipe(
    prompt=prompt, seed=1,
    infinityou_id_image=id_image, infinityou_guidance=1.0,
    controlnet_inputs=controlnet_inputs,
    num_inference_steps=50, embedded_guidance=3.5,
    height=height, width=width,
)
image.save("man.jpg")

prompt = "A woman, portrait, cinematic"
id_image = "data/examples/infiniteyou/woman.jpg"
id_image = Image.open(id_image).convert('RGB')
image = pipe(
    prompt=prompt, seed=1,
    infinityou_id_image=id_image, infinityou_guidance=1.0,
    controlnet_inputs=controlnet_inputs,
    num_inference_steps=50, embedded_guidance=3.5,
    height=height, width=width,
)
image.save("woman.jpg")