import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from PIL import Image


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
pipe.load_lora(pipe.dit, "models/train/Insert-Anything_lora/epoch-0.safetensors", alpha=1)

image = pipe(
    insert_anything_source_image=Image.open("data/diffsynth_example_dataset/flux/Insert-Anything/source_image.png").convert("RGB"),
    insert_anything_source_mask=Image.open("data/diffsynth_example_dataset/flux/Insert-Anything/source_mask.png").convert("L"),
    insert_anything_ref_image=Image.open("data/diffsynth_example_dataset/flux/Insert-Anything/ref_image.png").convert("RGB"),
    insert_anything_ref_mask=Image.open("data/diffsynth_example_dataset/flux/Insert-Anything/ref_mask.png").convert("L"),
    seed=666, embedded_guidance=30.0, num_inference_steps=50,
)
image.save("image_Insert-Anything_lora.jpg")
