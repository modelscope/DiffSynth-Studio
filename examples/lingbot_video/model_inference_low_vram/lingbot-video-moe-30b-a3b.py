import os
import json
import torch
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

inference_dir = os.path.join(os.path.dirname(__file__), "..", "model_inference")
with open(os.path.join(inference_dir, "prompts", "t2v_example_1.json"), "r", encoding="utf-8") as f:
    caption = json.load(f)

video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-moe-30b-a3b_low_vram.mp4", fps=15, quality=10)

input_video = VideoData("video_lingbot-video-moe-30b-a3b_low_vram.mp4", height=480, width=832)
video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    input_video=input_video, denoising_strength=0.7,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=1,
)
save_video(video, "video_lingbot-video-moe-30b-a3b_v2v_low_vram.mp4", fps=15, quality=10)
