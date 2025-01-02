from diffsynth import ModelManager, FluxImagePipeline, download_customized_models
from modelscope import dataset_snapshot_download
from examples.EntityControl.utils import visualize_masks
from PIL import Image
import torch


# download and load model
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev"])
model_manager.load_lora(
    download_customized_models(
        model_id="DiffSynth-Studio/Eligen",
        origin_file_path="model_bf16.safetensors",
        local_dir="models/lora/entity_control"
    ),
    lora_alpha=1
)
pipe = FluxImagePipeline.from_model_manager(model_manager)

# download and load mask images
dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern="data/examples/eligen/mask*")
masks = [Image.open(f"./data/examples/eligen/mask{i}.png") for i in range(1, 8)]

entity_prompts = ["A beautiful woman", "mirror", "necklace", "glasses", "earring", "white dress", "jewelry headpiece"]
global_prompt = "A beautiful woman wearing white dress, holding a mirror, with a warm light background;"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw"

# generate image
image = pipe(
    prompt=global_prompt,
    cfg_scale=3.0,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    embedded_guidance=3.5,
    seed=4,
    height=1024,
    width=1024,
    eligen_entity_prompts=entity_prompts,
    eligen_entity_masks=masks,
    enable_eligen_on_negative=False,
)
image.save(f"entity_control.png")
visualize_masks(image, masks, entity_prompts, f"entity_control_with_mask.png")
