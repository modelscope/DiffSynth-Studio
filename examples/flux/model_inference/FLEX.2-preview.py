import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.utils.controlnet import Annotator
import numpy as np
from PIL import Image


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ostris/Flex.2-preview", origin_file_pattern="Flex.2-preview.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)

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
