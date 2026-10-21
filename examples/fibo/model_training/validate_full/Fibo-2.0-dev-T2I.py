from diffsynth.pipelines.fibo import FiboPipeline, ModelConfig
from diffsynth.core import load_state_dict
import torch


pipe = FiboPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="processor/"),
    font_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="fonts/"),
)
state_dict = load_state_dict("./models/train/Fibo-2.0-dev-T2I_full/epoch-1.safetensors", torch_dtype=torch.bfloat16)
pipe.dit.load_state_dict(state_dict)
image = pipe(prompt="a corgi dog sitting on grass, looking at camera, happily", seed=0, num_inference_steps=50, cfg_scale=5.0, height=1024, width=1024)
image.save("Fibo-2.0-dev-T2I_full.jpg")
