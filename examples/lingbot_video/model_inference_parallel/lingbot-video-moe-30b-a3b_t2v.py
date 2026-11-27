# torchrun --standalone --nproc_per_node=8 examples/lingbot_video/model_inference_parallel/lingbot-video-moe-30b-a3b_t2v.py
import torch
import json
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig, CFGParallelConfig

pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern="*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses"), cfg=CFGParallelConfig()),
)

# --- Text-to-video -------------------------------------------------------------------
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="lingbot_video/lingbot-video-moe-30b-a3b_t2v/*",
)
# LingBot-Video is trained on structured-JSON captions, not free-form prose. This example
# runs on a released in-distribution caption; see the bottom for turning a brief idea into
# such a caption with the two-stage prompt rewriter.
with open("data/diffsynth_example_dataset/lingbot_video/lingbot-video-moe-30b-a3b_t2v/t2v_example_1.json", "r", encoding="utf-8") as f:
    caption = json.load(f)

video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
if dist.get_rank() == 0:
    save_video(video, "video_lingbot-video-moe-30b-a3b_t2v.mp4", fps=15, quality=10)

# --- Video-to-video ------------------------------------------------------------------
# denoising_strength < 1 keeps part of the input structure.
dist.barrier()
input_video = VideoData("video_lingbot-video-moe-30b-a3b_t2v.mp4", height=480, width=832)
video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    input_video=input_video, denoising_strength=0.7,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=1,
)
if dist.get_rank() == 0:
    save_video(video, "video_lingbot-video-moe-30b-a3b_v2v.mp4", fps=15, quality=10)
