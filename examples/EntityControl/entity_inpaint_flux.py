import torch
from diffsynth import ModelManager, FluxImagePipeline, download_customized_models, FluxImageLoraPipeline
from examples.EntityControl.utils import visualize_masks
import os
import json
from PIL import Image

# lora_path = download_customized_models(
#     model_id="DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1",
#     origin_file_path="merged_lora.safetensors",
#     local_dir="models/lora"
# )[0]

lora_path = '/root/model_bf16.safetensors'
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "t2i_models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "t2i_models/FLUX/FLUX.1-dev/text_encoder_2",
    "t2i_models/FLUX/FLUX.1-dev/ae.safetensors",
    "t2i_models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
model_manager.load_lora(lora_path, lora_alpha=1.)

pipe = FluxImagePipeline.from_model_manager(model_manager)

mask_dir = '/mnt/nas1/zhanghong/DiffSynth-Studio/workdirs/tmp_mask'
image_shape = 1024
guidance = 3.5
cfg = 3.0
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"
names = ['inpaint2']
seeds = [0]
use_seperated_negtive_prompt = False
for name, seed in zip(names, seeds):
    out_dir = f'workdirs/paper_app/inpaint/elc/{name}'
    os.makedirs(out_dir, exist_ok=True)
    cur_dir = os.path.join(mask_dir, name)
    metas = json.load(open(os.path.join(mask_dir, name, 'prompts.json')))
    inpaint_input = Image.open(os.path.join(cur_dir, 'input.png')).convert('RGB')
    prompt = metas['global_prompt']
    prompt = 'A person with a dog walking on the cloud. A rocket in the sky'
    mask_prompts = metas['mask_prompts']
    masks = [Image.open(os.path.join(mask_dir, name, f"{mask_idx}.png")).resize((image_shape, image_shape), resample=Image.NEAREST) for mask_idx in range(len(mask_prompts))]
    torch.manual_seed(seed)
    image = pipe(
        prompt=prompt,
        cfg_scale=cfg,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        embedded_guidance=guidance,
        height=image_shape,
        width=image_shape,
        entity_prompts=mask_prompts,
        entity_masks=masks,
        inpaint_input=inpaint_input,
        use_seperated_negtive_prompt=use_seperated_negtive_prompt,
    )
    use_sep = f'_sepneg' if use_seperated_negtive_prompt else ''
    visualize_masks(image, masks, mask_prompts, os.path.join(out_dir, f"{name}_{seed}{use_sep}.png"))
