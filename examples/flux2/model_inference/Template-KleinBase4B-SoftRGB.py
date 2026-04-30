from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch

pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
)
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-SoftRGB")],
)
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "R": 128/255,
        "G": 128/255,
        "B": 128/255
    }],
)
image.save("image_rgb_normal.jpg")
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "R": 208/255,
        "G": 185/255,
        "B": 138/255
    }],
)
image.save("image_rgb_warm.jpg")
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "R": 94/255,
        "G": 163/255,
        "B": 174/255
    }],
)
image.save("image_rgb_cold.jpg")
