from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch
from PIL import Image
import numpy as np

def load_template_pipeline(model_ids):
    template = TemplatePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[ModelConfig(model_id=model_id) for model_id in model_ids],
    )
    return template

# Base Model
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
# image = pipe(
#     prompt="A cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
# )
# image.save("image_base.jpg")

# template = load_template_pipeline(["DiffSynth-Studio/Template-KleinBase4B-Brightness"])
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{"scale": 0.7}],
#     negative_template_inputs = [{"scale": 0.5}]
# )
# image.save("image_Brightness_light.jpg")
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{"scale": 0.5}],
#     negative_template_inputs = [{"scale": 0.5}]
# )
# image.save("image_Brightness_normal.jpg")
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{"scale": 0.3}],
#     negative_template_inputs = [{"scale": 0.5}]
# )
# image.save("image_Brightness_dark.jpg")

# template = load_template_pipeline(["DiffSynth-Studio/Template-KleinBase4B-ControlNet"])
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone, bathed in bright sunshine.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{
#         "image": Image.open("data/assets/image_depth.jpg"),
#         "prompt": "A cat is sitting on a stone, bathed in bright sunshine.",
#     }],
#     negative_template_inputs = [{
#         "image": Image.open("data/assets/image_depth.jpg"),
#         "prompt": "",
#     }],
# )
# image.save("image_ControlNet_sunshine.jpg")
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone, surrounded by colorful magical particles.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{
#         "image": Image.open("data/assets/image_depth.jpg"),
#         "prompt": "A cat is sitting on a stone, surrounded by colorful magical particles.",
#     }],
#     negative_template_inputs = [{
#         "image": Image.open("data/assets/image_depth.jpg"),
#         "prompt": "",
#     }],
# )
# image.save("image_ControlNet_magic.jpg")

# template = load_template_pipeline(["DiffSynth-Studio/Template-KleinBase4B-Edit"])
# image = template(
#     pipe,
#     prompt="Put a hat on this cat.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{
#         "image": Image.open("data/assets/image_reference.jpg"),
#         "prompt": "Put a hat on this cat.",
#     }],
#     negative_template_inputs = [{
#         "image": Image.open("data/assets/image_reference.jpg"),
#         "prompt": "",
#     }],
# )
# image.save("image_Edit_hat.jpg")
# image = template(
#     pipe,
#     prompt="Make the cat turn its head to look to the right.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{
#         "image": Image.open("data/assets/image_reference.jpg"),
#         "prompt": "Make the cat turn its head to look to the right.",
#     }],
#     negative_template_inputs = [{
#         "image": Image.open("data/assets/image_reference.jpg"),
#         "prompt": "",
#     }],
# )
# image.save("image_Edit_head.jpg")

# template = load_template_pipeline(["DiffSynth-Studio/Template-KleinBase4B-Upscaler"])
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{
#         "image": Image.open("data/assets/image_lowres_512.jpg"),
#         "prompt": "A cat is sitting on a stone.",
#     }],
#     negative_template_inputs = [{
#         "image": Image.open("data/assets/image_lowres_512.jpg"),
#         "prompt": "",
#     }],
# )
# image.save("image_Upscaler_1.png")
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{
#         "image": Image.open("data/assets/image_lowres_100.jpg"),
#         "prompt": "A cat is sitting on a stone.",
#     }],
#     negative_template_inputs = [{
#         "image": Image.open("data/assets/image_lowres_100.jpg"),
#         "prompt": "",
#     }],
# )
# image.save("image_Upscaler_2.png")

# template = load_template_pipeline(["DiffSynth-Studio/Template-KleinBase4B-SoftRGB"])
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{
#         "R": 128/255,
#         "G": 128/255,
#         "B": 128/255
#     }],
# )
# image.save("image_rgb_normal.jpg")
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{
#         "R": 208/255,
#         "G": 185/255,
#         "B": 138/255
#     }],
# )
# image.save("image_rgb_warm.jpg")
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{
#         "R": 94/255,
#         "G": 163/255,
#         "B": 174/255
#     }],
# )
# image.save("image_rgb_cold.jpg")

# template = load_template_pipeline(["DiffSynth-Studio/Template-KleinBase4B-PandaMeme"])
# image = template(
#     pipe,
#     prompt="A meme with a sleepy expression.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{}],
#     negative_template_inputs = [{}],
# )
# image.save("image_PandaMeme_sleepy.jpg")
# image = template(
#     pipe,
#     prompt="A meme with a happy expression.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{}],
#     negative_template_inputs = [{}],
# )
# image.save("image_PandaMeme_happy.jpg")
# image = template(
#     pipe,
#     prompt="A meme with a surprised expression.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{}],
#     negative_template_inputs = [{}],
# )
# image.save("image_PandaMeme_surprised.jpg")

# template = load_template_pipeline(["DiffSynth-Studio/Template-KleinBase4B-Sharpness"])
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{"scale": 0.1}],
#     negative_template_inputs = [{"scale": 0.5}],
# )
# image.save("image_Sharpness_0.1.jpg")
# image = template(
#     pipe,
#     prompt="A cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{"scale": 0.8}],
#     negative_template_inputs = [{"scale": 0.5}],
# )
# image.save("image_Sharpness_0.8.jpg")

# template = load_template_pipeline(["DiffSynth-Studio/Template-KleinBase4B-Inpaint"])
# image = template(
#     pipe,
#     prompt="An orange cat is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{
#         "image": Image.open("data/assets/image_reference.jpg"),
#         "mask": Image.open("data/assets/image_mask_1.jpg"),
#         "force_inpaint": True,
#     }],
#     negative_template_inputs = [{
#         "image": Image.open("data/assets/image_reference.jpg"),
#         "mask": Image.open("data/assets/image_mask_1.jpg"),
#     }],
# )
# image.save("image_Inpaint_1.jpg")
# image = template(
#     pipe,
#     prompt="A cat wearing sunglasses is sitting on a stone.",
#     seed=0, cfg_scale=4, num_inference_steps=50,
#     template_inputs = [{
#         "image": Image.open("data/assets/image_reference.jpg"),
#         "mask": Image.open("data/assets/image_mask_2.jpg"),
#     }],
#     negative_template_inputs = [{
#         "image": Image.open("data/assets/image_reference.jpg"),
#         "mask": Image.open("data/assets/image_mask_2.jpg"),
#     }],
# )
# image.save("image_Inpaint_2.jpg")
