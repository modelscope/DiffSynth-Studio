import os
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig, normalize_caption


# Validate a TI2V LoRA (trained with lora/lingbot-video-dense-1.3b_ti2v.sh) by conditioning
# on a first frame via `input_image`, exactly as at inference. Same base T2V checkpoint.
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
pipe.load_lora(pipe.dit, "models/train/lingbot-video-dense-1.3b_ti2v_lora/epoch-19.safetensors", alpha=1)

# Reuse the released first frame + caption from the inference example.
inference_dir = os.path.join(os.path.dirname(__file__), "..", "..", "model_inference")
caption = normalize_caption(os.path.join(inference_dir, "prompts", "ti2v_example.json"))
input_image = Image.open(os.path.join(inference_dir, "assets", "ti2v_first_frame.png")).convert("RGB")

video = pipe(
    prompt=caption,
    input_image=input_image,
    height=480, width=832, num_frames=169,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-dense-1.3b_ti2v.mp4", fps=15, quality=10)
