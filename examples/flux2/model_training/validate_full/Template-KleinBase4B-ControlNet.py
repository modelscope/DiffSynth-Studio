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
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-ControlNet")],
)
state_dict = load_state_dict("./models/train/Template-KleinBase4B-ControlNet_full/epoch-1.safetensors", torch_dtype=torch.bfloat16)
template.models[0].load_state_dict(state_dict)
dataset_snapshot_download(
    "DiffSynth-Studio/examples_in_diffsynth",
    allow_file_pattern=["templates/*"],
    local_dir="data/examples",
)
image = template(
    pipe,
    prompt="A cat is sitting on a stone, bathed in bright sunshine.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "image": Image.open("data/examples/templates/image_depth.jpg"),
        "prompt": "A cat is sitting on a stone, bathed in bright sunshine.",
    }],
    negative_template_inputs = [{
        "image": Image.open("data/examples/templates/image_depth.jpg"),
        "prompt": "",
    }],
)
image.save("image_ControlNet_sunshine.jpg")
image = template(
    pipe,
    prompt="A cat is sitting on a stone, surrounded by colorful magical particles.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "image": Image.open("data/examples/templates/image_depth.jpg"),
        "prompt": "A cat is sitting on a stone, surrounded by colorful magical particles.",
    }],
    negative_template_inputs = [{
        "image": Image.open("data/examples/templates/image_depth.jpg"),
        "prompt": "",
    }],
)
image.save("image_ControlNet_magic.jpg")
