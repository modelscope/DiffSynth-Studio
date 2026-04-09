import random
import torch
from PIL import Image, ImageDraw, ImageFont
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download


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

def example(pipe, seeds, example_id, global_prompt, entity_prompts):
    dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern=f"data/examples/eligen/entity_control/example_{example_id}/*.png")
    masks = [Image.open(f"./data/examples/eligen/entity_control/example_{example_id}/{i}.png").convert('RGB') for i in range(len(entity_prompts))]
    negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"
    for seed in seeds:
        # generate image
        image = pipe(
            prompt=global_prompt,
            cfg_scale=3.0,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            embedded_guidance=3.5,
            seed=seed,
            height=1024,
            width=1024,
            eligen_entity_prompts=entity_prompts,
            eligen_entity_masks=masks,
        )
        image.save(f"eligen_example_{example_id}_{seed}.png")
        visualize_masks(image, masks, entity_prompts, f"eligen_example_{example_id}_mask_{seed}.png")


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)
pipe.load_lora(pipe.dit, ModelConfig(model_id="DiffSynth-Studio/Eligen", origin_file_pattern="model_bf16.safetensors"), alpha=1)

# example 1
global_prompt = "A breathtaking beauty of Raja Ampat by the late-night moonlight , one beautiful woman from behind wearing a pale blue long dress with soft glow, sitting at the top of a cliff looking towards the beach,pastell light colors, a group of small distant birds flying in far sky, a boat sailing on the sea, best quality, realistic, whimsical, fantastic, splash art, intricate detailed, hyperdetailed, maximalist style, photorealistic, concept art, sharp focus, harmony, serenity, tranquility, soft pastell colors,ambient occlusion, cozy ambient lighting, masterpiece, liiv1, linquivera, metix, mentixis, masterpiece, award winning, view from above\n"
entity_prompts = ["cliff", "sea", "moon", "sailing boat", "a seated beautiful woman", "pale blue long dress with soft glow"]
example(pipe, [0], 1, global_prompt, entity_prompts)

# example 2
global_prompt = "samurai girl wearing a kimono, she's holding a sword  glowing with red flame, her long hair is flowing in the wind, she is looking at a small bird perched on the back of her hand. ultra realist style. maximum image detail. maximum realistic render."
entity_prompts = ["flowing hair", "sword glowing with red flame", "A cute bird", "blue belt"]
example(pipe, [0], 2, global_prompt, entity_prompts)

# example 3
global_prompt = "Image of a neverending staircase up to a mysterious palace in the sky, The ancient palace stood majestically atop a mist-shrouded mountain, sunrise, two traditional monk walk in the stair looking at the sunrise, fog,see-through, best quality, whimsical, fantastic, splash art, intricate detailed, hyperdetailed, photorealistic, concept art, harmony, serenity, tranquility, ambient occlusion, halation, cozy ambient lighting, dynamic lighting,masterpiece, liiv1, linquivera, metix, mentixis, masterpiece, award winning,"
entity_prompts = ["ancient palace", "stone staircase with railings", "a traditional monk", "a traditional monk"]
example(pipe, [27], 3, global_prompt, entity_prompts)

# example 4
global_prompt = "A beautiful girl wearing shirt and shorts in the street,  holding a sign 'Entity Control'"
entity_prompts = ["A beautiful girl", "sign 'Entity Control'", "shorts", "shirt"]
example(pipe, [21], 4, global_prompt, entity_prompts)

# example 5
global_prompt = "A captivating, dramatic scene in a painting that exudes mystery and foreboding. A white sky, swirling blue clouds, and a crescent yellow moon illuminate a solitary woman standing near the water's edge. Her long dress flows in the wind, silhouetted against the eerie glow. The water mirrors the fiery sky and moonlight, amplifying the uneasy atmosphere."
entity_prompts = ["crescent yellow moon", "a solitary woman", "water", "swirling blue clouds"]
example(pipe, [0], 5, global_prompt, entity_prompts)

# example 6
global_prompt = "Snow White and the 6 Dwarfs."
entity_prompts = ["Dwarf 1", "Dwarf 2", "Dwarf 3", "Snow White", "Dwarf 4", "Dwarf 5", "Dwarf 6"]
example(pipe, [8], 6, global_prompt, entity_prompts)

# example 7, same prompt with different seeds
seeds = range(5, 9)
global_prompt = "A beautiful woman wearing white dress, holding a mirror, with a warm light background;"
entity_prompts = ["A beautiful woman", "mirror", "necklace", "glasses", "earring", "white dress", "jewelry headpiece"]
example(pipe, seeds, 7, global_prompt, entity_prompts)
