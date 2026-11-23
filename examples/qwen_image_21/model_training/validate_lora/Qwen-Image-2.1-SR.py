from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
import torch
from PIL import Image
from modelscope import snapshot_download

pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
)
pipe.load_lora(pipe.dit, "models/train/Qwen-Image-2.1-SR_lora/epoch-4.safetensors")
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-2.1-SR", origin_file_pattern="Template-DetailScale/")],
)

snapshot_download("DiffSynth-Studio/Qwen-Image-2.1-SR", allow_file_pattern="assets/example_*/*", local_dir="data")

prompt = "Restore the degraded x4 low-resolution image into a clean, natural and detailed high-resolution image. Remove degradation artifacts while recovering plausible fine details. Strictly preserve the original layout, geometry, colors, object identity and visible text. Do not add, delete, replace or relocate content."
edit_image = Image.open("data/assets/example_scale_1/input.png")
for detail_scale in [0.1, 0.5, 0.9]:
    image = template(
        pipe,
        prompt=prompt,
        edit_image=edit_image,
        height=edit_image.height,
        width=edit_image.width,
        seed=0,
        num_inference_steps=40,
        cfg_scale=2.0,
        template_inputs=[{"scale": detail_scale}],
        negative_template_inputs=[{"scale": 1 - detail_scale}],
    )
    image.save(f"image_SR_scale_{detail_scale}.png")
