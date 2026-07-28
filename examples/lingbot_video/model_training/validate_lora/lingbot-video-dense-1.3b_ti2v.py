import os
import json
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig


# TI2V LoRA validation: load the fresh LoRA checkpoint and generate from the same caption
# + first frame the TI2V inference example uses. Adjust `epoch-N.safetensors` to whichever
# epoch you want to validate (matches the .sh's `--num_epochs`).
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
pipe.load_lora(pipe.dit, "models/train/lingbot-video-dense-1.3b_ti2v_lora/epoch-4.safetensors", alpha=1)

# Reuse the released first frame + caption from the inference example.
inference_dir = os.path.join(os.path.dirname(__file__), "..", "..", "model_inference")
with open(os.path.join(inference_dir, "prompts", "ti2v_example.json"), "r", encoding="utf-8") as f:
    caption = json.load(f)
input_image = Image.open(os.path.join(inference_dir, "assets", "ti2v_first_frame.png")).convert("RGB")

video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    input_image=input_image,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-dense-1.3b_ti2v.mp4", fps=15, quality=10)
