from diffsynth.pipelines.krea2 import Krea2Pipeline, ModelConfig
import torch


pipe = Krea2Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # For LoRA models trained on Krea-2-Raw, we recommend using them on Krea-2-Turbo.
        ModelConfig(model_id="krea/Krea-2-Raw", origin_file_pattern="raw.safetensors"),
        ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern="*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
)
pipe.load_lora(pipe.dit, "models/train/Krea-2-Raw_lora/epoch-4.safetensors")
prompt = "A dog"
image = pipe(prompt, seed=0, num_inference_steps=52, cfg_scale=4.5)
image.save("image.jpg")
