from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
import torch
from PIL import Image
from modelscope import snapshot_download
from diffsynth.core.quant import QuantizeConfig

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
    "quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw"),
}
vram_config_cpu = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors", **vram_config_cpu),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
pipe.load_lora(pipe.dit, ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-2.1-SR", origin_file_pattern="sr_lora.safetensors"))
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-2.1-SR", origin_file_pattern="Template-DetailScale/")],
    lazy_loading=True,
)

snapshot_download("DiffSynth-Studio/Qwen-Image-2.1-SR", allow_file_pattern="assets/example_*/*", local_dir="data")

prompt = "Restore the degraded x4 low-resolution image into a clean, natural and detailed high-resolution image. Remove degradation artifacts while recovering plausible fine details. Strictly preserve the original layout, geometry, colors, object identity and visible text. Do not add, delete, replace or relocate content."

# Super-resolution with the default detail scale
edit_image = Image.open("data/assets/example_noise_1/input.png")
image = template(
    pipe,
    prompt=prompt,
    edit_image=edit_image,
    height=edit_image.height,
    width=edit_image.width,
    seed=0,
    num_inference_steps=40,
    cfg_scale=2.0,
    template_inputs=[{"scale": 0.9}],
    negative_template_inputs=[{"scale": 0.1}],
)
image.save("image_SR_1.png")

# The same input restored with different detail scales. The content is preserved and
# only the amount of high-frequency structure changes.
edit_image = Image.open("data/assets/example_scale_1/input.png")
image = template(
    pipe,
    prompt=prompt,
    edit_image=edit_image,
    height=edit_image.height,
    width=edit_image.width,
    seed=0,
    num_inference_steps=40,
    cfg_scale=2.0,
    template_inputs=[{"scale": 0.9}],
    negative_template_inputs=[{"scale": 0.1}],
)
image.save("image_SR_scale_0.9.png")
