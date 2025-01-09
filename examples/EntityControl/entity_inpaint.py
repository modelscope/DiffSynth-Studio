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
dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern="data/examples/eligen/inpaint/*")
masks = [Image.open(f"./data/examples/eligen/inpaint/inpaint_mask_{i}.png") for i in range(1, 3)]
input_image = Image.open("./data/examples/eligen/inpaint/inpaint_image.jpg")

entity_prompts = ["A person wear red shirt", "Airplane"]
global_prompt = "A person walking on the path in front of a house; An airplane in the sky"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw, blur"

# generate image
image = pipe(
    prompt=global_prompt,
    input_image=input_image,
    cfg_scale=3.0,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    embedded_guidance=3.5,
    seed=0,
    height=1024,
    width=1024,
    eligen_entity_prompts=entity_prompts,
    eligen_entity_masks=masks,
    enable_eligen_on_negative=False,
    enable_eligen_inpaint=True,
)
image.save(f"entity_inpaint.png")
visualize_masks(image, masks, entity_prompts, f"entity_inpaint_with_mask.png")
