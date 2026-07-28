import os
import json
import torch
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig


# Text-to-image (t2i). t2i is text-to-video with a single frame: pass num_frames=1 through
# the same pipeline and DiT (no separate image weight). The only image-specific knob is the
# negative prompt -- `pipe.default_negative_prompt_image` drops the temporal/motion terms
# that cannot apply to a still frame. The pipeline returns a 1-frame list, i.e. one PIL
# image.

pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="processor/"),
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
frames[0].save("image_lingbot-video-dense-1.3b_t2i.png")
