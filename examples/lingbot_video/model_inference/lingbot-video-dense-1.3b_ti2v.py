import os
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig, normalize_caption


# Image-to-video (TI2V). Dense-1.3B reuses the SAME T2V checkpoint — there is no separate
# i2v weight. The condition first frame is used twice: as visual input to the Qwen3-VL text
# encoder, and as a clean latent pinned into the first frame of the diffusion latent so the
# model only generates the frames that follow. Pass a first frame via `input_image`.

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

# prompts/ti2v_example.json is a released in-distribution caption paired with the first frame
# in assets/ti2v_first_frame.png. The caption should describe the motion that unfolds from the
# given frame; the pipeline calls normalize_caption internally so a path also works directly.
here = os.path.dirname(__file__)
caption = normalize_caption(os.path.join(here, "prompts", "ti2v_example.json"))
input_image = Image.open(os.path.join(here, "assets", "ti2v_first_frame.png")).convert("RGB")

video = pipe(
    prompt=caption,
    input_image=input_image,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-dense-1.3b_ti2v.mp4", fps=15, quality=10)
