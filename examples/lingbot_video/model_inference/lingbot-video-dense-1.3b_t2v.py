import torch
import json
from diffsynth.utils.data import save_video, VideoData
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

# --- Text-to-video -------------------------------------------------------------------
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="lingbot_video/lingbot-video-dense-1.3b_t2v/*",
)
# LingBot-Video is trained on structured-JSON captions, not free-form prose. This example
# runs on a released in-distribution caption; see the bottom for turning a brief idea into
# such a caption with the two-stage prompt rewriter.
with open("data/diffsynth_example_dataset/lingbot_video/lingbot-video-dense-1.3b_t2v/t2v_example_1.json", "r", encoding="utf-8") as f:
    caption = json.load(f)

video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-dense-1.3b_t2v.mp4", fps=15, quality=10)

# --- Video-to-video ------------------------------------------------------------------
# denoising_strength < 1 keeps part of the input structure.
input_video = VideoData("video_lingbot-video-dense-1.3b_t2v.mp4", height=480, width=832)
video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    input_video=input_video, denoising_strength=0.7,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=1,
)
save_video(video, "video_lingbot-video-dense-1.3b_v2v.mp4", fps=15, quality=10)

# --- Optional: rewrite a brief idea into a structured caption ------------------------
# The two-stage rewriter (model_training/scripts/prompt_rewriter.py) is a separate VLM +
# stage-2 LoRA adapter (NOT the DiT) and is not downloaded automatically. Fetch both
# weights and point the env vars at them:
#     modelscope download --model Qwen/Qwen3.6-27B --local_dir ./models/Qwen/Qwen3.6-27B
#     modelscope download --model Robbyant/lingbot-video-rewriter-lora --local_dir ./models/Robbyant/lingbot-video-rewriter-lora
#     export REWRITER_BASE_MODEL=./models/Qwen/Qwen3.6-27B
#     export REWRITER_ADAPTER=./models/Robbyant/lingbot-video-rewriter-lora
#
# from examples.lingbot_video.model_training.scripts.prompt_rewriter import rewrite_prompt
# caption = rewrite_prompt(
#     "A playful puppy runs across a lush green meadow, chasing a red ball. "
#     "Dynamic side-tracking camera.",
#     mode="t2v", duration=5,
# )
# video = pipe(prompt=caption, height=480, width=832, num_frames=81, cfg_scale=3.0, seed=0)
# save_video(video, "video_lingbot-video-dense-1.3b_rewrite.mp4", fps=15, quality=10)
