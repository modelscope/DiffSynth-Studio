import torch, json
from PIL import Image, ImageDraw
from diffsynth.pipelines.fibo import FiboPipeline, ModelConfig

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
pipe = FiboPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="processor/"),
    font_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="fonts/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Text-to-image
caption = {
    "short_description": "An origami whale swimming through paper clouds.",
    "objects": [{"description": "a blue origami whale, folded from glossy paper", "location": "center"}],
    "background_setting": "a bright sky made of layered white paper clouds",
    "lighting": {"conditions": "soft daylight", "direction": "from above"},
    "style_medium": "paper craft photograph",
}
prompt = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
image = pipe(prompt=prompt, height=1024, width=1024, seed=42, num_inference_steps=50, cfg_scale=5.0)
image.save("fibo_2_dev_t2i.png")

# Edit the generated image
edited = pipe(
    prompt="change the lighting to a warm sunset, keep the origami whale and paper clouds",
    edit_image=image, seed=43, num_inference_steps=50, cfg_scale=5.0,
)
edited.save("fibo_2_dev_edit.png")

# Masked edit of the generated image
mask = Image.new("L", image.size, 0)
ImageDraw.Draw(mask).ellipse((image.width // 4, image.height // 3, 3 * image.width // 4, 2 * image.height // 3), fill=255)
mask.save("fibo_2_dev_mask.png")
masked = pipe(
    prompt="add a red paper crane on the origami whale",
    edit_image=image, mask=mask, seed=44, num_inference_steps=50, cfg_scale=5.0,
)
masked.save("fibo_2_dev_edit_mask.png")

# Text rendering: the caption's `text_render` entries are drawn as small images and shown to Qwen3-VL
fonts = pipe.default_fonts
caption = {
    "short_description": "A young woman holding a white sign that reads exactly \"ModelScope\".",
    "objects": [
        {"description": "a young woman with long dark hair, wearing a light blue sweater, smiling", "location": "center"},
        {"description": "a white rectangular sign held with both hands, displaying exactly the characters ModelScope in bold black text", "location": "center"},
    ],
    "background_setting": "a clean, softly lit studio with a pale gradient background",
    "lighting": {"conditions": "soft studio lighting", "direction": "front"},
    "photographic_characteristics": {"depth_of_field": "shallow", "focus": "sharp on the sign"},
    "style_medium": "photograph",
    "text_render": [{"text": "ModelScope"}],
}
prompt = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
image = pipe(
    prompt=prompt, height=1024, width=1024, seed=45, num_inference_steps=50, cfg_scale=5.0,
    fonts=fonts,
)
image.save("fibo_2_dev_text_render.png")
