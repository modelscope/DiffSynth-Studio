import torch
from diffsynth import ModelManager, FluxImagePipeline, download_customized_models
from examples.EntityControl.utils import visualize_masks
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
seed = 4
# set True to apply regional attention in negative prompt prediction for better results with more time
use_seperated_negtive_prompt = False
mask_urls = [
    'https://github.com/user-attachments/assets/02905f6e-40c2-4482-9abe-b1ce50ccabbf',
    'https://github.com/user-attachments/assets/a4cf4361-abf7-4556-ba94-74683eda4cb7',
    'https://github.com/user-attachments/assets/b6595ff4-7269-4d8f-acf0-5df40bd6c59f',
    'https://github.com/user-attachments/assets/941d39a7-3aa1-437f-8b2a-4adb15d2fb3e',
    'https://github.com/user-attachments/assets/400c4086-5398-4291-b1b5-22d8483c08d9',
    'https://github.com/user-attachments/assets/ce324c77-fa1d-4aad-a5cb-698f0d5eca70',
    'https://github.com/user-attachments/assets/4e62325f-a60c-44f7-b53b-6da0869bb9db'
]
# prepare entity masks, entity prompts, global prompt and negative prompt
masks = []
for url in mask_urls:
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).resize((image_shape, image_shape), resample=Image.NEAREST)
    masks.append(mask)
entity_prompts = ["A beautiful woman", "mirror", "necklace", "glasses", "earring", "white dress", "jewelry headpiece"]
global_prompt = "A beautiful woman wearing white dress, holding a mirror, with a warm light background;"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw"

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
    use_seperated_negtive_prompt=use_seperated_negtive_prompt,
)
image.save(f"entity_control.png")
visualize_masks(image, masks, entity_prompts, f"entity_control_with_mask.png")
