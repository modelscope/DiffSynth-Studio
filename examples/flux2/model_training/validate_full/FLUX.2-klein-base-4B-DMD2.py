from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
import torch

pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:2",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
)
state_dict = load_state_dict("./models/train/FLUX.2-klein-base-4B_dmd2-4steps/step-200.safetensors", torch_dtype=torch.bfloat16)
pipe.dit.load_state_dict(state_dict)

prompt = "a dog"
image = pipe(prompt=prompt, seed=0, num_inference_steps=4, height=512, width=512)
image.save("image_FLUX.2-klein-base-4B-DMD2-4steps.jpg")