from pathlib import Path


def latest_checkpoint(directory):
    checkpoints = list(Path(directory).glob("epoch-*.safetensors"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {directory}")
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("-", 1)[-1]))


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
pipe.load_lora(pipe.dit, str(latest_checkpoint('./models/train/Krea-2-Raw_split')))
prompt = "A dog"
image = pipe(prompt, seed=0, num_inference_steps=52, cfg_scale=4.5)
image.save('split_training_Krea-2-Raw.jpg')
