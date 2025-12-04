from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
from PIL import Image, ImageDraw, ImageFont
from modelscope import dataset_snapshot_download, snapshot_download
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
        font = ImageFont.truetype("wqy-zenhei.ttc", font_size)  # Adjust as needed
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


def example(pipe, seeds, example_id, global_prompt, entity_prompts, height=784, width=1280):
    dataset_snapshot_download(
        dataset_id="DiffSynth-Studio/examples_in_diffsynth",
        local_dir="./",
        allow_file_pattern=f"data/examples/eligen/poster/example_{example_id}/*.png"
    )
    masks = [
        Image.open(f"./data/examples/eligen/poster/example_{example_id}/{i}.png").convert('RGB').resize((width, height))
        for i in range(len(entity_prompts))
    ]
    negative_prompt = "网格化，规则的网格，模糊, 低分辨率, 低质量, 变形, 畸形, 错误的解剖学, 变形的手, 变形的身体, 变形的脸, 变形的头发, 变形的眼睛, 变形的嘴巴"
    for seed in seeds:
        # generate image
        image = pipe(
            prompt=global_prompt,
            cfg_scale=4.0,
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            seed=seed,
            height=height,
            width=width,
            eligen_entity_prompts=entity_prompts,
            eligen_entity_masks=masks,
        )
        image.save(f"eligen_poster_example_{example_id}_{seed}.png")
        image = Image.new("RGB", (width, height), (0, 0, 0))
        visualize_masks(image, masks, entity_prompts, f"eligen_poster_example_{example_id}_mask_{seed}.png")


vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
snapshot_download(
    "DiffSynth-Studio/Qwen-Image-EliGen-Poster",
    local_dir="models/DiffSynth-Studio/Qwen-Image-EliGen-Poster",
    allow_file_pattern="model.safetensors",
)
pipe.load_lora(pipe.dit, "models/DiffSynth-Studio/Qwen-Image-EliGen-Poster/model.safetensors", hotload=True)
global_prompt = "一张以柔粉紫为背景的海报，左侧有大号粉紫色文字“Qwen-Image EliGen-Poster”，粉紫色椭圆框内白色小字：“图像精确分区控制模型”。右侧有一只小兔子在拆礼物，旁边站着一只头顶迷你烟花发射器的小龙（卡通Q版）。背景有一些白云点缀。整体风格卡通可爱，传达节日惊喜的主题。"
entity_prompts = ["粉紫色文字“Qwen-Image EliGen-Poster”", "粉紫色椭圆框内白色小字：“图像精确分区控制模型”", "一只小兔子在拆礼物，小兔子旁边站着一只头顶迷你烟花发射器的小龙（卡通Q版）"]
seed = [42]
example(pipe, seed, 1, global_prompt, entity_prompts)
