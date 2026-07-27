import os
import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig, normalize_caption


# Low-VRAM inference. Setting `offload_dtype` / `offload_device` on each ModelConfig is
# what actually turns on DiffSynth's VRAM management: weights are kept on CPU in fp8 and
# streamed to the GPU layer-by-layer, then computed in bf16. `vram_limit` on its own has
# no effect — without a non-None `offload_dtype` and `offload_device` the loader never
# enables offloading (see `need_to_enable_vram_management`). `vram_limit` only sets how
# much resident VRAM the streaming may use once offloading is on.
vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
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
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

# Use a released in-distribution structured caption (shared with the model_inference
# example). LingBot-Video is trained on structured-JSON captions, not free-form prose.
caption = normalize_caption(os.path.join(
    os.path.dirname(__file__), "..", "model_inference", "prompts", "t2v_example_1.json"))
video = pipe(
    prompt=caption,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-dense-1.3b_low_vram.mp4", fps=15, quality=10)
