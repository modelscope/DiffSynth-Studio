import torch
from diffsynth.pipelines.flux3_image import Flux3ImagePipeline, ModelConfig
from diffsynth.core import load_state_dict

checkpoint_path = "workspace/your_training_run/step-1000.safetensors"

pipe = Flux3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein-base", origin_file_pattern="model.safetensors"),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/"),
)

state_dict = load_state_dict(checkpoint_path, torch_dtype=torch.bfloat16)
pipe.dit.load_state_dict(state_dict, strict=False)

prompt = "dog,white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt, height=1024, width=1024, seed=0, num_inference_steps=8, cfg_scale=1.0)
image.save("image_flux3_dmd2_full_8steps.jpg")
