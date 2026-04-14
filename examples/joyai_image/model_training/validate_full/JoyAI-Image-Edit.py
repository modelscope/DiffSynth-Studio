import torch
from PIL import Image
from diffsynth.pipelines.joyai_image import JoyAIImagePipeline, ModelConfig
from diffsynth import load_state_dict

pipe = JoyAIImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="transformer/transformer.pth"),
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="JoyAI-Image-Und/model*.safetensors"),
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="vae/Wan2.1_VAE.pth"),
    ],
    processor_config=ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="JoyAI-Image-Und/"),
)

state_dict = load_state_dict("models/train/JoyAI-Image-Edit_full/epoch-1.safetensors")
pipe.dit.load_state_dict(state_dict)

prompt = "将裙子改为粉色"
edit_image = Image.open("data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/edit/image1.jpg").convert("RGB")

image = pipe(
    prompt=prompt,
    edit_image=edit_image,
    height=1024,
    width=1024,
    seed=0,
    num_inference_steps=50,
    cfg_scale=5.0,
)
image.save("image_full.jpg")
