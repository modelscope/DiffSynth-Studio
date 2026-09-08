import torch
from safetensors.torch import load_file
from diffsynth.pipelines.sensenova_u1_image import SenseNovaU1ImagePipeline, ModelConfig


pipe = SenseNovaU1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT-SFT", origin_file_pattern="model*.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT-SFT", origin_file_pattern="./"),
)

state_dict = load_file("./models/train/SenseNova-U1.5-8B-MoT-SFT_full/epoch-1.safetensors", device="cuda")
pipe.dit.load_state_dict(state_dict, strict=False)

image = pipe(
    prompt="dog,white and brown dog, sitting on wall, under pink flowers",
    seed=42,
    height=1024,
    width=1024,
    num_inference_steps=50,
    cfg_scale=4.0,
    shift=3.0,
)
image.save("image.jpg")
