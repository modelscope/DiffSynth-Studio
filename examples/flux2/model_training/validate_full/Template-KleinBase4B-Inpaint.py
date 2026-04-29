from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
import torch
from modelscope import dataset_snapshot_download
from PIL import Image

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
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-Inpaint")],
)
state_dict = load_state_dict("./models/train/Template-KleinBase4B-Inpaint_full/epoch-1.safetensors", torch_dtype=torch.bfloat16)
template.models[0].load_state_dict(state_dict)
dataset_snapshot_download(
    "DiffSynth-Studio/examples_in_diffsynth",
    allow_file_pattern=["templates/*"],
    local_dir="data/examples",
)
image = template(
    pipe,
    prompt="An orange cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "image": Image.open("data/examples/templates/image_reference.jpg"),
        "mask": Image.open("data/examples/templates/image_mask_1.jpg"),
        "force_inpaint": True,
    }],
    negative_template_inputs = [{
        "image": Image.open("data/examples/templates/image_reference.jpg"),
        "mask": Image.open("data/examples/templates/image_mask_1.jpg"),
    }],
)
image.save("image_Inpaint_1.jpg")
image = template(
    pipe,
    prompt="A cat wearing sunglasses is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "image": Image.open("data/examples/templates/image_reference.jpg"),
        "mask": Image.open("data/examples/templates/image_mask_2.jpg"),
    }],
    negative_template_inputs = [{
        "image": Image.open("data/examples/templates/image_reference.jpg"),
        "mask": Image.open("data/examples/templates/image_mask_2.jpg"),
    }],
)
image.save("image_Inpaint_2.jpg")

