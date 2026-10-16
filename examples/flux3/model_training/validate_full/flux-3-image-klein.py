from diffsynth.pipelines.flux3_image import Flux3ImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
import torch


pipe = Flux3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="model.safetensors"),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/"),
)
state_dict = load_state_dict("./models/train/flux-3-image-klein_full/epoch-1.safetensors", torch_dtype=torch.bfloat16)
pipe.dit.load_state_dict(state_dict)
prompt = "a dog"
image = pipe(prompt=prompt, seed=0, num_inference_steps=16, cfg_scale=1.0)
image.save("image.jpg")
