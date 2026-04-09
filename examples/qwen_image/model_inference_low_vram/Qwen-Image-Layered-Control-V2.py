from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image
import torch


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
        ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Layered-Control", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.dit, ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Layered-Control-V2", origin_file_pattern="model.safetensors", **vram_config))

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="layer_v2/*.png"
)

prompt = "Text 'APRIL'"
input_image = Image.open("data/example_image_dataset/layer_v2/image_1.png").convert("RGBA").resize((1024, 1024))
image = pipe(
    prompt, seed=0,
    height=1024, width=1024,
    layer_input_image=input_image, layer_num=0,
    num_inference_steps=10, cfg_scale=4,
)
image[0].save("image_prompt.png")

mask_image = Image.open("data/example_image_dataset/layer_v2/mask_2.png").convert("RGBA").resize((1024, 1024))
input_image = Image.open("data/example_image_dataset/layer_v2/image_2.png").convert("RGBA").resize((1024, 1024))
image = pipe(
    prompt, seed=0,
    height=1024, width=1024,
    layer_input_image=input_image, layer_num=0,
    context_image=mask_image,
    num_inference_steps=10, cfg_scale=1.0,
)
image[0].save("image_mask.png")
