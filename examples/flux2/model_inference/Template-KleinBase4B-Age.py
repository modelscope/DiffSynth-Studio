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
image = template(
    pipe,
    prompt="A portrait of a woman with black hair, wearing a suit.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"age": 20}],
    negative_template_inputs=[{"age": 45}],
)
image.save(f"image_age_20.jpg")
image = template(
    pipe,
    prompt="A portrait of a woman with black hair, wearing a suit.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"age": 50}],
    negative_template_inputs=[{"age": 45}],
)
image.save(f"image_age_50.jpg")
image = template(
    pipe,
    prompt="A portrait of a woman with black hair, wearing a suit.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"age": 80}],
    negative_template_inputs=[{"age": 45}],
)
image.save(f"image_age_80.jpg")
