from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image
import torch


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

dataset_snapshot_download(
    "DiffSynth-Studio/example_image_dataset",
    allow_patterns="layer/image.png",
    local_dir="data/example_image_dataset"
)

# Prompt should be provided to the pipeline. Our pipeline will not generate the prompt.
prompt = 'A cheerful child with brown hair is waving enthusiastically under a bright blue sky filled with colorful confetti and balloons. The word "HELLO!" is prominently displayed in bold red letters above the child, while "Have a Great Day!" appears in elegant cursive at the bottom right corner. The scene is vibrant and festive, with a mix of pastel colors and dynamic shapes creating a joyful atmosphere.'
# Height and width should be consistent with input_image and be divided evenly by 16
input_image = Image.open("data/example_image_dataset/layer/image.png").convert("RGBA").resize((864, 480))
images = pipe(
    prompt,
    seed=1, num_inference_steps=50,
    height=480, width=864,
    layer_input_image=input_image, layer_num=3,
)
for i, image in enumerate(images):
    if i == 0: continue # The first image is the input image.
    image.save(f"image_{i}.png")
