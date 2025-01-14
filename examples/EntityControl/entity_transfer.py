from diffsynth import ModelManager, FluxImagePipeline, download_customized_models
from modelscope import dataset_snapshot_download
from examples.EntityControl.utils import visualize_masks
from PIL import Image
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


def generate(pipe: FluxImagePipeline, source_image, target_image, mask, height, width, prompt, entity_prompt, image_save_path, mask_save_path, seed=0):
    input_mask = Image.new('RGB', (width * 2, height))
    input_mask.paste(mask.resize((width, height), resample=Image.NEAREST).convert('RGB'), (width, 0))

    input_image = Image.new('RGB', (width * 2, height))
    input_image.paste(source_image.resize((width, height)).convert('RGB'), (0, 0))
    input_image.paste(target_image.resize((width, height)).convert('RGB'), (width, 0))

    image = pipe(
        prompt=prompt,
        input_image=input_image,
        cfg_scale=3.0,
        negative_prompt="",
        num_inference_steps=50,
        embedded_guidance=3.5,
        seed=seed,
        height=height,
        width=width * 2,
        eligen_entity_prompts=[entity_prompt],
        eligen_entity_masks=[input_mask],
        enable_eligen_on_negative=False,
        enable_eligen_inpaint=True,
    )
    target_image = image.crop((width, 0, 2 * width, height))
    target_image.save(image_save_path)
    visualize_masks(target_image, [mask], [entity_prompt], mask_save_path)
    return target_image


pipe = build_pipeline()

dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern="data/examples/eligen/logo_transfer/*")

prompt="The two-panel image showcases the joyful identity, with the left panel showing a rabbit graphic; [LEFT] while the right panel translates the design onto a shopping tote with the rabbit logo in black, held by a person in a market setting, emphasizing the brand's approachable and eco-friendly vibe."
logo_prompt="a rabbit logo"

logo_image = Image.open("data/examples/eligen/logo_transfer/source_image.png")
target_image = Image.open("data/examples/eligen/logo_transfer/target_image.png")
mask = Image.open("data/examples/eligen/logo_transfer/mask_1.png")
generate(
    pipe, logo_image, target_image, mask, 
    height=1024, width=1024,
    prompt=prompt, entity_prompt=logo_prompt,
    image_save_path="entity_transfer_1.png",
    mask_save_path="entity_transfer_with_mask_1.png"
)

mask = Image.open("data/examples/eligen/logo_transfer/mask_2.png")
generate(
    pipe, logo_image, target_image, mask, 
    height=1024, width=1024,
    prompt=prompt, entity_prompt=logo_prompt,
    image_save_path="entity_transfer_2.png",
    mask_save_path="entity_transfer_with_mask_2.png"
)
