from diffsynth.pipelines.flux3_image import Flux3ImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
import torch


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
state_dict = load_state_dict("./models/train/flux-3-image-klein-base_full/epoch-1.safetensors", torch_dtype=torch.bfloat16)
pipe.dit.load_state_dict(state_dict)
prompt = "a dog"
image = pipe(prompt=prompt, seed=0, num_inference_steps=50, cfg_scale=3.5)
image.save("image.jpg")
