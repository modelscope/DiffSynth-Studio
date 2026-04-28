from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
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
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-PandaMeme")],
)
state_dict = load_state_dict("./models/train/Template-KleinBase4B-PandaMeme_full/epoch-1.safetensors", torch_dtype=torch.bfloat16)
template.models[0].load_state_dict(state_dict)
image = template(
    pipe,
    prompt="A meme with a sleepy expression.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{}],
    negative_template_inputs = [{}],
)
image.save("image_PandaMeme_sleepy.jpg")
image = template(
    pipe,
    prompt="A meme with a happy expression.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{}],
    negative_template_inputs = [{}],
)
image.save("image_PandaMeme_happy.jpg")
image = template(
    pipe,
    prompt="A meme with a surprised expression.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{}],
    negative_template_inputs = [{}],
)
image.save("image_PandaMeme_surprised.jpg")
