import torch
from diffsynth import load_state_dict
from diffsynth.pipelines.boogu_image import BooguImagePipeline, ModelConfig

pipe = BooguImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Turbo", origin_file_pattern="mllm/*.safetensors"),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Turbo", origin_file_pattern="vae/*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Boogu/Boogu-Image-0.1-Turbo", origin_file_pattern="mllm/"),
)

state_dict = load_state_dict("models/train/Boogu-Image-0.1-Turbo_full/epoch-1.safetensors")
pipe.dit.load_state_dict(state_dict, strict=False)

prompt = "dog,white and brown dog, sitting on wall, under pink flowers"

output = pipe(
    prompt=prompt,
    negative_prompt="",
    height=1024,
    width=1024,
    seed=42,
    rand_device="cuda",
    num_inference_steps=4,
    cfg_scale=1.0,
    sigmas=[0.999, 0.748, 0.5, 0.25],
)
output.save("image_Boogu-Image-0.1-Turbo_full.jpg")
