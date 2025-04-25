import torch
from diffsynth import ModelManager, FluxImagePipeline, download_models
from diffsynth.controlnets.processors import Annotator
import numpy as np
from PIL import Image


download_models(["FLUX.1-dev"])
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/ostris/Flex.2-preview/Flex.2-preview.safetensors"
])
pipe = FluxImagePipeline.from_model_manager(model_manager)

image = pipe(
    prompt="portrait of a beautiful Asian girl, long hair, red t-shirt, sunshine, beach",
    num_inference_steps=50, embedded_guidance=3.5,
    seed=0
)
image.save("image_1.jpg")

mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
mask[200:400, 400:700] = 255
mask = Image.fromarray(mask)
mask.save("image_mask.jpg")

inpaint_image = image

image = pipe(
    prompt="portrait of a beautiful Asian girl with sunglasses, long hair, red t-shirt, sunshine, beach",
    num_inference_steps=50, embedded_guidance=3.5,
    flex_inpaint_image=inpaint_image, flex_inpaint_mask=mask,
    seed=4
)
image.save("image_2.jpg")

control_image = Annotator("canny")(image)
control_image.save("image_control.jpg")

image = pipe(
    prompt="portrait of a beautiful Asian girl with sunglasses, long hair, yellow t-shirt, sunshine, beach",
    num_inference_steps=50, embedded_guidance=3.5,
    flex_control_image=control_image,
    seed=4
)
image.save("image_3.jpg")
