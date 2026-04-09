from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig, ControlNetInput
from modelscope import dataset_snapshot_download
from PIL import Image
import torch


vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1", origin_file_pattern="Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Control
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="depth/image_1.jpg"
)
controlnet_image = Image.open("data/example_image_dataset/depth/image_1.jpg").resize((1024, 1024))
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt=prompt, seed=0, height=1024, width=1024, controlnet_inputs=[ControlNetInput(image=controlnet_image, scale=0.7)])
image.save("image_control.jpg")

# Inpaint
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="inpaint/*.jpg"
)
inpaint_image = Image.open("./data/example_image_dataset/inpaint/image_1.jpg").convert("RGB").resize((1024, 1024))
inpaint_mask = Image.open("./data/example_image_dataset/inpaint/mask.jpg").convert("RGB").resize((1024, 1024))
prompt = "一只戴着墨镜的猫"
image = pipe(prompt=prompt, seed=0, height=1024, width=1024, controlnet_inputs=[ControlNetInput(inpaint_image=inpaint_image, inpaint_mask=inpaint_mask, scale=0.7)])
image.save("image_inpaint.jpg")
