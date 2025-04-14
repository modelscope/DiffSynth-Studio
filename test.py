import torch
from diffsynth import ModelManager, FluxImagePipeline, download_models, load_state_dict
from diffsynth.models.flux_reference_embedder import FluxReferenceEmbedder
from PIL import Image


model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
pipe = FluxImagePipeline.from_model_manager(model_manager)

pipe.reference_embedder = FluxReferenceEmbedder().to(dtype=torch.bfloat16, device="cuda")
pipe.reference_embedder.init()

for i in range(4):
    image = pipe(
        prompt="a girl.",
        num_inference_steps=30, embedded_guidance=3.5,
        height=512, width=512,
        reference_images=[Image.open("data/example4.jpg").resize((512, 512))]
    )
    image.save(f"image_{i}.jpg")