import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from PIL import Image
from modelscope import dataset_snapshot_download

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="flux1-fill-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Redux-dev", origin_file_pattern="image_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Redux-dev", origin_file_pattern="image_embedder/diffusion_pytorch_model.safetensors"),
    ],
)

pipe.load_lora(pipe.dit, ModelConfig(model_id="HuanJue/Insert-Anything", origin_file_pattern="20250321_steps5000_pytorch_lora_weights.safetensors"))

dataset_snapshot_download(
    dataset_id="HuanJue/example_dataset",
    local_dir="./",
    allow_file_pattern=f"Insert-Anything/*",
)

source_image = Image.open("Insert-Anything/source_image.png").convert("RGB")
source_mask = Image.open("Insert-Anything/source_mask.png").convert("L")
ref_image = Image.open("Insert-Anything/ref_image.png").convert("RGB")
ref_mask = Image.open("Insert-Anything/ref_mask.png").convert("L")

seed = 666

image = pipe(
    insert_anything_source_image=source_image,
    insert_anything_source_mask=source_mask,
    insert_anything_ref_image=ref_image,
    insert_anything_ref_mask=ref_mask,
    seed=seed,
    embedded_guidance=30.0,
    num_inference_steps=50,
)

image.save("image_Insert-Anything.jpg")
