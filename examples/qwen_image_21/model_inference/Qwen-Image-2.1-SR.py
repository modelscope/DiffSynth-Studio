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
pipe.load_lora(pipe.dit, ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-2.1-SR", origin_file_pattern="sr_lora.safetensors"))
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-2.1-SR", origin_file_pattern="Template-DetailScale/")],
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

# Text-rich scenes. The prompt describes the content that should be recovered.
edit_image = Image.open("data/assets/example_text_1/input.png")
image = template(
    pipe,
    prompt="一张展现中国传统二十四节气“谷雨”的春日风景海报。画面中青山绿水，粉色的樱花盛开且花瓣随风飘落，草地上站着一位女孩和一只小狗正眺望着蜿蜒的河流，空中还有一只飞燕，整体充满生机与春意。海报上方印有醒目的黑色书法大字“谷雨”及红色小印章，下方配有英文“Grain Rain”和“24 solar terms”；画面中部竖排写着诗句“看百谷丛生，听春雨绵绵”；底部则标注了日期“[04.20]”以及中英文对照的“中国传统二十四节气”与“Chinese traditional 24 solar terms”。",
    edit_image=edit_image,
    height=edit_image.height,
    width=edit_image.width,
    seed=0,
    num_inference_steps=40,
    cfg_scale=2.0,
    template_inputs=[{"scale": 0.9}],
    negative_template_inputs=[{"scale": 0.1}],
)
image.save("image_SR_text.png")

# The detail control module also works for text-to-image generation
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    height=1024,
    width=1024,
    seed=0,
    num_inference_steps=40,
    cfg_scale=2.0,
    template_inputs=[{"scale": 0.9}],
    negative_template_inputs=[{"scale": 0.1}],
)
image.save("image_T2I_scale_0.9.png")
