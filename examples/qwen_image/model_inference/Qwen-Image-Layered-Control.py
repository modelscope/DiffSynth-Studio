from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import snapshot_download
from PIL import Image
import torch


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Layered-Control", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

snapshot_download(
    model_id="DiffSynth-Studio/Qwen-Image-Layered-Control",
    allow_file_pattern="assets/image_1_input.png",
    local_dir="data/layered_input"
)

prompt = "A cartoon skeleton character wearing a purple hat and holding a gift box"
input_image = Image.open("data/layered_input/assets/image_1_input.png").convert("RGBA").resize((1024, 1024))
images = pipe(
    prompt,
    seed=0,
    num_inference_steps=30, cfg_scale=4,
    height=1024, width=1024,
    layer_input_image=input_image,
    layer_num=0,
)
images[0].save("image.png")
