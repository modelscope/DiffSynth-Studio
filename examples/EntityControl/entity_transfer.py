from diffsynth import ModelManager, FluxImagePipeline, download_customized_models
from diffsynth.data.video import crop_and_resize
from modelscope import dataset_snapshot_download
from examples.EntityControl.utils import visualize_masks
from PIL import Image
import numpy as np
import torch



def build_pipeline():
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev"])
    model_manager.load_lora(
        download_customized_models(
            model_id="DiffSynth-Studio/Eligen",
            origin_file_path="model_bf16.safetensors",
            local_dir="models/lora/entity_control"
        ),
        lora_alpha=1
    )
    model_manager.load_lora(
        download_customized_models(
            model_id="iic/In-Context-LoRA",
            origin_file_path="visual-identity-design.safetensors",
            local_dir="models/lora/In-Context-LoRA"
        ),
        lora_alpha=1
    )
    pipe = FluxImagePipeline.from_model_manager(model_manager)
    return pipe


def generate(pipe: FluxImagePipeline, logo_image, target_image, mask, height, width, prompt, logo_prompt, image_save_path, mask_save_path):
    mask = Image.fromarray(np.concatenate([
        np.ones((height, width, 3), dtype=np.uint8) * 0,
        np.array(crop_and_resize(mask, height, width)),
    ], axis=1))

    input_image = Image.fromarray(np.concatenate([
        np.array(crop_and_resize(logo_image, height, width)),
        np.array(crop_and_resize(target_image, height, width)),
    ], axis=1))

    image = pipe(
        prompt=prompt,
        input_image=input_image,
        cfg_scale=3.0,
        negative_prompt="",
        num_inference_steps=50,
        embedded_guidance=3.5,
        seed=0,
        height=height,
        width=width * 2,
        eligen_entity_prompts=[logo_prompt],
        eligen_entity_masks=[mask],
        enable_eligen_on_negative=False,
        enable_eligen_inpaint=True,
    )
    image.save(image_save_path)
    visualize_masks(image, [mask], [logo_prompt], mask_save_path)


pipe = build_pipeline()

dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern="data/examples/eligen/logo_transfer/*")
logo_image = Image.open("data/examples/eligen/logo_transfer/logo_transfer_logo.png")
target_image = Image.open("data/examples/eligen/logo_transfer/logo_transfer_target_image.png")

prompt="The two-panel image showcases the joyful identity, with the left panel showing a rabbit graphic; [LEFT] while the right panel translates the design onto a shopping tote with the rabbit logo in black, held by a person in a market setting, emphasizing the brand's approachable and eco-friendly vibe."
logo_prompt="a rabbit logo"

mask = Image.open("data/examples/eligen/logo_transfer/logo_transfer_mask_1.png")
generate(
    pipe, logo_image, target_image, mask, 
    height=1024, width=736,
    prompt=prompt, logo_prompt=logo_prompt,
    image_save_path="entity_transfer_1.png",
    mask_save_path="entity_transfer_with_mask_1.png"
)

mask = Image.open("data/examples/eligen/logo_transfer/logo_transfer_mask_2.png")
generate(
    pipe, logo_image, target_image, mask, 
    height=1024, width=736,
    prompt=prompt, logo_prompt=logo_prompt,
    image_save_path="entity_transfer_2.png",
    mask_save_path="entity_transfer_with_mask_2.png"
)
