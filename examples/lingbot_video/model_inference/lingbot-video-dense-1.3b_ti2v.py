import os
import json
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig


# Image-to-video (TI2V). Dense-1.3B reuses the SAME T2V checkpoint -- there is no separate
# i2v weight. The condition first frame is used twice: as visual input to the Qwen3-VL text
# encoder, and VAE-encoded to a clean latent pinned into the first frame of the diffusion
# latent (and re-pinned after every denoising step) so the model only generates the frames
# that follow. Pass a first frame via `input_image`.

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

# In-tree released ti2v caption + paired first frame; the reviewer will move these to the
# diffsynth_example_dataset repo in a follow-up pass.
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
save_video(video, "video_lingbot-video-dense-1.3b_ti2v.mp4", fps=15, quality=10)
