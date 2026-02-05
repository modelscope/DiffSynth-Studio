from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import snapshot_download
from PIL import Image
import torch

# Load models
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
lora = ModelConfig(
    model_id="DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA",
    origin_file_pattern="model.safetensors"
)
pipe.load_lora(pipe.dit, lora)

# Load images
snapshot_download(
    "DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA",
    local_dir="./data",
    allow_file_pattern="assets/*"
)
edit_image = [
    Image.open("data/assets/image1_original.png"),
    Image.open("data/assets/image1_edit_1.png"),
    Image.open("data/assets/image2_original.png")
]
prompt = "Edit image 3 based on the transformation from image 1 to image 2."
negative_prompt = "泛黄，AI感，不真实，丑陋，油腻的皮肤，异常的肢体，不协调的肢体"

# Generate
image_4 = pipe(
    prompt=prompt, negative_prompt=negative_prompt,
    edit_image=edit_image,
    seed=1,
    num_inference_steps=50,
    height=1280,
    width=720,
    zero_cond_t=True,
)
image_4.save("image.png")