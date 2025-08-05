from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
from PIL import Image, ImageDraw, ImageFont
from modelscope import dataset_snapshot_download
import random


def visualize_masks(image, masks, mask_prompts, output_path, font_size=35, use_random_colors=False):
    # Create a blank image for overlays
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    
    colors = [
        (165, 238, 173, 80),
        (76, 102, 221, 80),
        (221, 160, 77, 80),
        (204, 93, 71, 80),
        (145, 187, 149, 80),
        (134, 141, 172, 80),
        (157, 137, 109, 80),
        (153, 104, 95, 80),
        (165, 238, 173, 80),
        (76, 102, 221, 80),
        (221, 160, 77, 80),
        (204, 93, 71, 80),
        (145, 187, 149, 80),
        (134, 141, 172, 80),
        (157, 137, 109, 80),
        (153, 104, 95, 80),
    ]
    # Generate random colors for each mask
    if use_random_colors:
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 80) for _ in range(len(masks))]
    
    # Font settings
    try:
        font = ImageFont.truetype("arial", font_size)  # Adjust as needed
    except IOError:
        font = ImageFont.load_default(font_size)

    # Overlay each mask onto the overlay image
    for mask, mask_prompt, color in zip(masks, mask_prompts, colors):
        # Convert mask to RGBA mode
        mask_rgba = mask.convert('RGBA')
        mask_data = mask_rgba.getdata()
        new_data = [(color if item[:3] == (255, 255, 255) else (0, 0, 0, 0)) for item in mask_data]
        mask_rgba.putdata(new_data)

        # Draw the mask prompt text on the mask
        draw = ImageDraw.Draw(mask_rgba)
        mask_bbox = mask.getbbox()  # Get the bounding box of the mask
        text_position = (mask_bbox[0] + 10, mask_bbox[1] + 10)  # Adjust text position based on mask position
        draw.text(text_position, mask_prompt, fill=(255, 255, 255, 255), font=font)

        # Alpha composite the overlay with this mask
        overlay = Image.alpha_composite(overlay, mask_rgba)
    
    # Composite the overlay onto the original image
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    
    # Save or display the resulting image
    result.save(output_path)

    return result


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
example_id = 1
global_prompt = "A breathtaking beauty of Raja Ampat by the late-night moonlight , one beautiful woman from behind wearing a long dress, sitting at the top of a cliff looking towards the beach,pastell light colors, a group of small distant birds flying in far sky, a boat sailing on the sea\n"
dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern=f"data/examples/eligen/entity_control/example_{example_id}/*.png")
entity_prompts = ["cliff", "sea", "red moon", "sailing boat", "a seated beautiful woman wearing red dress", "yellow long dress"]
masks = [Image.open(f"./data/examples/eligen/entity_control/example_{example_id}/{i}.png").convert('RGB') for i in range(len(entity_prompts))]

for seed in range(20):
    image = pipe(global_prompt, seed=seed, num_inference_steps=40, eligen_entity_prompts=entity_prompts, eligen_entity_masks=masks, cfg_scale=4.0, height=1024, width=1024)
    image.save(f"workdirs/qwen_image/eligen_{seed}.jpg")

    visualize_masks(image, masks, entity_prompts, f"workdirs/qwen_image/eligen_{seed}_mask.png")

    image1 = pipe(global_prompt, seed=seed, num_inference_steps=40, height=1024, width=1024, cfg_scale=4.0)
    image1.save(f"workdirs/qwen_image/qwenimage_{seed}.jpg")
