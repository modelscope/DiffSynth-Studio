from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
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
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
state_dict = load_state_dict("models/train/Qwen-Image-Layered_full/epoch-1.safetensors")
pipe.dit.load_state_dict(state_dict)
prompt = "a poster"
input_image = Image.open("data/example_image_dataset/layer/image.png").convert("RGBA").resize((864, 480))
images = pipe(
    prompt, seed=0,
    height=480, width=864,
    layer_input_image=input_image, layer_num=3,
)
for i, image in enumerate(images):
    if i == 0: continue # The first image is the input image.
    image.save(f"image_{i}.png")
