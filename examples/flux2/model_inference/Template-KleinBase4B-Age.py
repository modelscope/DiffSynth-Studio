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
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-Age")],
)
prompt = "Half body color photograph of a single woman, head and torso with visible arms and hands resting gently in front of the body, looking directly at the camera, centered composition, colorful studio background with soft gradient of warm pastel tones, vibrant studio lighting, wearing a plain red short-sleeve t-shirt, straight black shoulder-length hair, photorealistic, high quality"# prompt = "Full body photograph of a single woman standing, looking directly at the camera, centered composition, plain neutral gray background, soft even studio lighting, wearing a plain white short-sleeve t-shirt and blue jeans, barefoot, arms resting naturally at sides, straight black shoulder-length hair, photorealistic, high quality"
negative_age = 45
for age in range(10, 91, 5):
    print(f"Generating age {age}...")
    image = template(
        pipe,
        prompt=prompt,
        seed=0, cfg_scale=4, num_inference_steps=50,
        template_inputs=[{"age": age}],
        negative_template_inputs=[{"age": negative_age}],
    )
    image.save(f"image_age_{age}.jpg")
