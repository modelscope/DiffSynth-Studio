import torch
from PIL import Image, ImageDraw
from diffsynth.pipelines.fibo import FiboPipeline, ModelConfig
from modelscope import dataset_snapshot_download

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
        ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="processor/"),
    font_config=ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="fonts/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="fibo/Fibo-2.0-dev-Edit/image_1.png"
)
image = Image.open("data/DiffSynth-Studio/diffsynth_example_dataset/fibo/Fibo-2.0-dev-Edit/image_1.png")

# Edit the generated image
edited = pipe(
    prompt="change the lighting to a warm sunset, keep the origami whale and paper clouds",
    edit_image=image, seed=43, num_inference_steps=4, cfg_scale=1.0,
)
edited.save("fibo_2_turbo_edit.png")

# Masked edit of the generated image
mask = Image.new("L", image.size, 0)
ImageDraw.Draw(mask).ellipse((image.width // 4, image.height // 3, 3 * image.width // 4, 2 * image.height // 3), fill=255)
mask.save("fibo_2_dev_mask.png")
masked = pipe(
    prompt="add a red paper crane on the origami whale",
    edit_image=image, mask=mask, seed=44, num_inference_steps=4, cfg_scale=1.0,
)
masked.save("fibo_2_turbo_edit_mask.png")
