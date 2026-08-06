import os
import json
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig
from diffsynth import load_state_dict
from modelscope import dataset_snapshot_download

pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern="*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
)
state_dict = load_state_dict("models/train/lingbot-video-dense-1.3b_ti2v_full/epoch-1.safetensors")
pipe.dit.load_state_dict(state_dict)

# The condition first frame and its paired caption ship in the example dataset.
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="lingbot_video/lingbot-video-dense-1.3b_ti2v/*",
)
base = "data/diffsynth_example_dataset/lingbot_video/lingbot-video-dense-1.3b_ti2v"
with open(os.path.join(base, "ti2v_example.json"), "r", encoding="utf-8") as f:
    caption = json.load(f)
input_image = Image.open(os.path.join(base, "ti2v_first_frame.png")).convert("RGB")

video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    input_image=input_image,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-dense-1.3b_ti2v.mp4", fps=15, quality=10)
