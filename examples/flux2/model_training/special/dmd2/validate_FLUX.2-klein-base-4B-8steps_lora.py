import torch
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

lora_checkpoint = "workspace/your_training_run/step-1000.safetensors"

pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
)

pipe.load_lora(pipe.dit, lora_checkpoint)

prompt = "dog,white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt, height=1024, width=1024, seed=0, num_inference_steps=8, cfg_scale=1.0)
image.save("image_flux2_dmd2_lora_8steps.jpg")
