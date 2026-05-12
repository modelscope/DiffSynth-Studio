"""HiDream-O1-Image Dev model Text-to-Image inference example."""
import torch
from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline
from diffsynth.core.loader.config import ModelConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

pipe = HiDreamO1ImagePipeline.from_pretrained(
    model_configs=[
        ModelConfig(
            model_id="HiDream-ai/HiDream-O1-Image-Dev",
            origin_file_pattern="model-*.safetensors",
        ),
    ],
    tokenizer_config=ModelConfig(
        model_id="HiDream-ai/HiDream-O1-Image-Dev",
        origin_file_pattern="tokenizer_config.json",
    ),
    torch_dtype=torch.bfloat16,
    device=device,
)

image = pipe(
    prompt="medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room.",
    negative_prompt=" ",
    cfg_scale=0.0,  # Dev model: no guidance
    height=2048,
    width=2048,
    seed=42,
    num_inference_steps=28,
    shift=1.0,
    noise_scale_start=7.5,
    noise_scale_end=7.5,
    noise_clip_std=2.5,
)
import os
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_hidream_o1_image_dev.png")
image.save(output_path)
print(f"Saved to {output_path}")
