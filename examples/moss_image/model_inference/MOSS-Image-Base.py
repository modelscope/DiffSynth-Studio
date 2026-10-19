from diffsynth.pipelines.moss_image import MOSSImagePipeline, ModelConfig
import torch
import json

pipe = MOSSImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", computation_dtype=torch.float32),
    ],
    tokenizer_config=ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="tokenizer/"),
)

prompt = '''A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:
"记得买牛奶和鸡蛋！
牛奶两盒，鸡蛋一打。
顺便带一袋面包。"'''

image = pipe(prompt, height=1024, width=1024, seed=42, num_inference_steps=50, cfg_scale=3.0)
image.save("image_MOSS-Image-Base.jpg")


# layout control
layout = {
    "summary": {
        "meta": ["poster"],
        "canvas": "single",
        "caption": "A minimalist travel poster with a cream background, a blue mountain illustration, and the title EXPLORE at the top.",
    },
    "items": [
        {
            "label": "text",
            "caption": "Large dark-blue uppercase title centered at the top.",
            "text": "EXPLORE",
            "bbox": [-416, -384, -288, 384],
        },
        {
            "label": "object",
            "caption": "A flat blue mountain illustration in the lower center.",
            "bbox": [-192, -288, 384, 288],
        },
    ],
    "bbox_coord_space": "abs_centralized",
    "bbox_format": "yxyx",
}
image = pipe(json.dumps(layout, ensure_ascii=False), height=1024, width=1024, seed=42, num_inference_steps=50, cfg_scale=3.0)
image.save("image_MOSS-Image-Base-Layout.jpg")