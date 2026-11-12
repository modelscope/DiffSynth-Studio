import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig
from PIL import Image

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Comfy-Org/MiniMax-H3-Image", origin_file_pattern="diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors", **vram_config),
        ModelConfig(model_id="Comfy-Org/MiniMax-H3", origin_file_pattern="text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

prompt = (
    "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough "
    "on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts, "
    "warm afternoon light raking across the grain of the wood."
)
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image-Int8-Pruned-Comfy.jpg")

prompt = "Change the color of the tomato in <Picture 1> to green."
image = pipe(
    prompt=prompt, height=4096, width=4096, seed=43,
    edit_image=[Image.open("MiniMax-H3-Image-Int8-Pruned-Comfy.jpg")],
)
image.save("MiniMax-H3-Image-Int8-Pruned-Comfy-Edit.jpg")
