import os
import json
import torch
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

with open(os.path.join(os.path.dirname(__file__), "prompts", "t2i_example.json"), "r", encoding="utf-8") as f:
    caption = json.load(f)

frames = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt_image,
    height=480, width=832, num_frames=1,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
frames[0].save("image_lingbot-video-moe-30b-a3b_t2i.png")
