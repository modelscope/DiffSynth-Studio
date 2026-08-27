from pathlib import Path


def latest_checkpoint(directory):
    checkpoints = list(Path(directory).glob("epoch-*.safetensors"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {directory}")
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("-", 1)[-1]))


import torch
import json
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig
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
pipe.load_lora(pipe.dit, str(latest_checkpoint('./models/train/lingbot-video-dense-1.3b_t2v_split')), alpha=1)
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="lingbot_video/lingbot-video-dense-1.3b_t2v/*",
)
with open("data/diffsynth_example_dataset/lingbot_video/lingbot-video-dense-1.3b_t2v/t2v_example_1.json", "r", encoding="utf-8") as f:
    caption = json.load(f)

video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, 'split_training_lingbot-video-dense-1.3b_t2v.mp4', fps=15, quality=10)
