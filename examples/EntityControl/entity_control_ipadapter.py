import torch
from diffsynth import ModelManager, FluxImagePipeline, download_customized_models
from examples.EntityControl.utils import visualize_masks
from PIL import Image
import requests
from io import BytesIO

lora_path = download_customized_models(
    model_id="DiffSynth-Studio/Eligen",
    origin_file_path="model_bf16.safetensors",
    local_dir="models/lora/entity_control"
)[0]
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev", "InstantX/FLUX.1-dev-IP-Adapter"])
model_manager.load_lora(lora_path, lora_alpha=1.)
pipe = FluxImagePipeline.from_model_manager(model_manager)

# prepare inputs
image_shape = 1024
seed = 4
# set True to apply regional attention in negative prompt prediction for better results with more time
use_seperated_negtive_prompt = False
mask_urls = [
    'https://github.com/user-attachments/assets/e6745b3f-ab2b-4612-9bb5-b7235474a9a4',
    'https://github.com/user-attachments/assets/5ddf9a89-32fa-4540-89ad-e956130942b3',
    'https://github.com/user-attachments/assets/9d8a0bb0-6817-497e-af85-44f2512afe79'
]
# prepare entity masks, entity prompts, global prompt and negative prompt
masks = []
for url in mask_urls:
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).resize((image_shape, image_shape), resample=Image.NEAREST)
    masks.append(mask)
entity_prompts = ['A girl', 'hat', 'sunset']
global_prompt = "A girl wearing a hat, looking at the sunset"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw"

response = requests.get('https://github.com/user-attachments/assets/019bbfaa-04b3-4de6-badb-32b67c29a1bc')
reference_img = Image.open(BytesIO(response.content)).convert('RGB').resize((image_shape, image_shape))

torch.manual_seed(seed)
image = pipe(
    prompt=global_prompt,
    cfg_scale=3.0,
    negative_prompt=negative_prompt,
    num_inference_steps=50, embedded_guidance=3.5, height=image_shape, width=image_shape,
    entity_prompts=entity_prompts, entity_masks=masks,
    use_seperated_negtive_prompt=use_seperated_negtive_prompt,
    ipadapter_images=[reference_img], ipadapter_scale=0.7
)
image.save(f"styled_entity_control.png")
visualize_masks(image, masks, entity_prompts, f"styled_entity_control_with_mask.png")
