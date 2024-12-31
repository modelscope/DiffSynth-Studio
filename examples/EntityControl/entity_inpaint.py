import torch
from diffsynth import ModelManager, FluxImagePipeline, download_customized_models, FluxImageLoraPipeline
from examples.EntityControl.utils import visualize_masks
import os
import json
from PIL import Image
import requests
from io import BytesIO

# download and load model
lora_path = download_customized_models(
    model_id="DiffSynth-Studio/Eligen",
    origin_file_path="model_bf16.safetensors",
    local_dir="models/lora/entity_control"
)[0]
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev"])
model_manager.load_lora(lora_path, lora_alpha=1.)
pipe = FluxImagePipeline.from_model_manager(model_manager)

# prepare inputs
image_shape = 1024
seed = 0
# set True to apply regional attention in negative prompt prediction for better results with more time
use_seperated_negtive_prompt = False
mask_urls = [
    'https://github.com/user-attachments/assets/0cf78663-5314-4280-a065-31ded7a24a46',
    'https://github.com/user-attachments/assets/bd3938b8-72a8-4d56-814f-f6445971b91d'
]
# prepare entity masks, entity prompts, global prompt and negative prompt
masks = []
for url in mask_urls:
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).resize((image_shape, image_shape), resample=Image.NEAREST)
    masks.append(mask)
entity_prompts = ["A person wear red shirt", "Airplane"]
global_prompt = "A person walking on the path in front of a house; An airplane in the sky"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw, blur"

response = requests.get('https://github.com/user-attachments/assets/fa4d6ba5-08fd-4fc7-adbb-19898d839364')
inpaint_input = Image.open(BytesIO(response.content)).convert('RGB').resize((image_shape, image_shape))

# generate image
torch.manual_seed(seed)
image = pipe(
    prompt=global_prompt,
    cfg_scale=3.0,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    embedded_guidance=3.5,
    height=image_shape,
    width=image_shape,
    entity_prompts=entity_prompts,
    entity_masks=masks,
    inpaint_input=inpaint_input,
    use_seperated_negtive_prompt=use_seperated_negtive_prompt,
)
image.save(f"entity_inpaint.png")
visualize_masks(image, masks, entity_prompts, f"entity_inpaint_with_mask.png")