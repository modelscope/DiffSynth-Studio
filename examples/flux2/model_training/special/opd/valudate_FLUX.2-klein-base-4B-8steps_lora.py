from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch

pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.dit, "./models/train/FLUX.2-klein-base-4B_opd_lora/step-1000.safetensors", alpha=1)
prompt = "dog,white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt=prompt, seed=0, num_inference_steps=8, cfg_scale=1, height=768, width=768)
image.save("image.jpg")
