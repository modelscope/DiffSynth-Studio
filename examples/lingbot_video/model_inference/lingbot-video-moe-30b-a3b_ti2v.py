import os
import json
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig

pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="processor/"),
)

here = os.path.dirname(__file__)
with open(os.path.join(here, "prompts", "ti2v_example.json"), "r", encoding="utf-8") as f:
    caption = json.load(f)
input_image = Image.open(os.path.join(here, "assets", "ti2v_first_frame.png")).convert("RGB")

video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    input_image=input_image,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-moe-30b-a3b_ti2v.mp4", fps=15, quality=10)
