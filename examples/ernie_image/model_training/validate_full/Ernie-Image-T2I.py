import torch
from diffsynth.pipelines.ernie_image import ErnieImagePipeline, ModelConfig
from diffsynth.core import load_state_dict

pipe = ErnieImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
)

state_dict = load_state_dict("./models/train/Ernie-Image-T2I_full/epoch-1.safetensors")
pipe.dit.load_state_dict(state_dict)

image = pipe(
    prompt="a professional photo of a cute dog",
    seed=0,
    num_inference_steps=50,
    cfg_scale=4.0,
)
image.save("image_full.jpg")
print("Full validation image saved to image_full.jpg")
