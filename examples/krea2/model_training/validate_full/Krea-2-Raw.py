from diffsynth.pipelines.krea2 import Krea2Pipeline, ModelConfig
import torch
from diffsynth import load_state_dict


pipe = Krea2Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="krea/Krea-2-Raw", origin_file_pattern="raw.safetensors"),
        ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern="*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
)
pipe.dit.load_state_dict(load_state_dict("models/train/Krea-2-Raw_full/epoch-1.safetensors"))
prompt = "A dog"
image = pipe(prompt, seed=0, num_inference_steps=52, cfg_scale=4.5)
image.save("image.jpg")
