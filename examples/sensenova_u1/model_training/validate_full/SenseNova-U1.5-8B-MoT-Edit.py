import torch
from PIL import Image
from safetensors.torch import load_file
from diffsynth.pipelines.sensenova_u1_image import SenseNovaU1ImagePipeline, ModelConfig


pipe = SenseNovaU1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="model*.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="./"),
)

state_dict = load_file("./models/train/SenseNova-U1.5-8B-MoT-Edit_full/epoch-1.safetensors", device="cuda")
pipe.dit.load_state_dict(state_dict, strict=False)

edit_image = Image.open("data/diffsynth_example_dataset/sensenova_u1/SenseNova-U1.5-8B-MoT-Edit/edit/image1.jpg").convert("RGB")

image = pipe(
    prompt="将裙子改为粉色",
    edit_image=edit_image,
    seed=42,
    height=1024,
    width=1024,
    num_inference_steps=50,
    cfg_scale=4.0,
    shift=3.0,
)
image.save("image.jpg")
