import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig


pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack", origin_file_pattern="models/dit_7bit.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/EntroPackPreQuants", origin_file_pattern="models/MiniMax-H3/text_encoder_4bpp.safetensors"),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="pdd_fused_heads.safetensors"),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
)

pipe.load_lora(pipe.dit, "./models/train/MiniMax-H3-Image-EntroPack_lora/epoch-4.safetensors")

prompt = "a dog"
image = pipe(prompt=prompt, height=1024, width=1024, seed=0)
image.save("MiniMax-H3-Image-EntroPack_lora.jpg")
