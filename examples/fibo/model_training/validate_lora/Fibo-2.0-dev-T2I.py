from diffsynth.pipelines.fibo import FiboPipeline, ModelConfig
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
pipe.load_lora(pipe.dit, "./models/train/Fibo-2.0-dev-T2I_lora/epoch-4.safetensors")
image = pipe(prompt="a corgi dog sitting on grass, looking at camera, happily", seed=0, num_inference_steps=50, cfg_scale=5.0, height=1024, width=1024)
image.save("Fibo-2.0-dev-T2I_lora.jpg")
