from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
import torch

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
state_dict = load_state_dict("./models/train/FLUX.2-klein-base-4B_opd_full/step-1000.safetensors", torch_dtype=torch.bfloat16)
pipe.dit.load_state_dict(state_dict)
prompt = "dog,white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt=prompt, seed=0, num_inference_steps=8, cfg_scale=1, height=768, width=768)
image.save("image.jpg")
