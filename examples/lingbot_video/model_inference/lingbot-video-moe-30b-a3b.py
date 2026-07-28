import torch
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig


# The MoE checkpoint is sharded, so the file pattern has to match all shards.
# 30B total parameters with 3B active per token (128 experts, top-8 routing).
pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="processor/"),
)

# --- Text-to-video -------------------------------------------------------------------
video = pipe(
    prompt="A playful puppy runs across a lush green meadow, its golden fur shining in the bright sunlight, ears perked up, chasing after a red ball. Wildflowers dot the grass, and a clear blue sky with a few white clouds stretches out behind it. Dynamic side-tracking camera.",
    negative_prompt=pipe.default_negative_prompt,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-moe-30b-a3b.mp4", fps=15, quality=10)

# --- Video-to-video ------------------------------------------------------------------
# denoising_strength < 1 keeps part of the input structure.
input_video = VideoData("video_lingbot-video-moe-30b-a3b.mp4", height=480, width=832)
video = pipe(
    prompt="A playful puppy wearing black sunglasses runs across a lush green meadow, its golden fur shining in the bright sunlight. Wildflowers dot the grass, and a clear blue sky with a few white clouds stretches out behind it. Dynamic side-tracking camera.",
    negative_prompt=pipe.default_negative_prompt,
    input_video=input_video, denoising_strength=0.7,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=1,
)
save_video(video, "video_lingbot-video-moe-30b-a3b_v2v.mp4", fps=15, quality=10)
