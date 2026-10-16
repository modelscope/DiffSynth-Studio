from diffsynth.pipelines.flux3_image import Flux3ImagePipeline, ModelConfig
import torch


pipe = Flux3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/flux-3-image-klein-entropack", origin_file_pattern="models/dit_4bit.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/flux-3-image-klein-entropack", origin_file_pattern="models/text_encoder_4bit.safetensors"),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/"),
)
pipe.load_lora(pipe.dit, "./models/train/flux-3-image-klein-entropack_lora/epoch-4.safetensors")
prompt = "a dog"
image = pipe(prompt=prompt, seed=0, num_inference_steps=16, cfg_scale=1.0)
image.save("image.jpg")
