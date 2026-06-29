from diffsynth.pipelines.krea2 import Krea2Pipeline, ModelConfig
import torch


pipe = Krea2Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="krea/Krea-2-Turbo", origin_file_pattern="turbo.safetensors"),
        ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern="*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
)
prompt = "Portrait of a woman in a blue dress, underwater, surrounded by colorful bubbles."
image = pipe(
    prompt, seed=0,
    height=2048, width=2048,
    # The following parameters are fixed.
    num_inference_steps=8, cfg_scale=1, mu=1.15,
)
image.save("image.jpg")
