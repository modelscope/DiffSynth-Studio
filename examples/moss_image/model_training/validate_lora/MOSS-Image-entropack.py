import torch
from diffsynth.pipelines.moss_image import MOSSImagePipeline, ModelConfig

# Base: replace MOSS-Image-SFT with MOSS-Image-Base.
pipe = MOSSImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/MOSS-Image-EntroPack", origin_file_pattern="models/MOSS-Image-SFT/dit_6bit.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/MOSS-Image-EntroPack", origin_file_pattern="models/text_encoder_6bit.safetensors"),
        ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", computation_dtype=torch.float32),
    ],
    tokenizer_config=ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="tokenizer/"),
)

pipe.load_lora(pipe.dit, "./models/train/MOSS-Image-entropack_lora/epoch-4.safetensors")

prompt = "a dog"
image = pipe(prompt, height=1024, width=1024, seed=0, num_inference_steps=50, cfg_scale=3.0)
image.save("image_lora.jpg")
