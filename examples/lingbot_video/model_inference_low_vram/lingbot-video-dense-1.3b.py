import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig


# Low-VRAM inference. `vram_limit` turns on DiffSynth's automatic VRAM management,
# which keeps model weights on CPU and streams them to the GPU layer-by-layer during
# each forward. The value below reserves ~2 GB of headroom on top of the models.
pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

video = pipe(
    prompt="A playful puppy runs across a lush green meadow, its golden fur shining in the bright sunlight, ears perked up, chasing after a red ball. Wildflowers dot the grass, and a clear blue sky with a few white clouds stretches out behind it. Dynamic side-tracking camera.",
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-dense-1.3b_low_vram.mp4", fps=15, quality=10)
