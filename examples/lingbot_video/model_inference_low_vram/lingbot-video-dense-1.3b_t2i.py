import os
import json
import torch
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig


# Low-VRAM text-to-image (t2i). t2i is text-to-video with num_frames=1 through the same
# pipeline and DiT (no separate image weight); the only image-specific knob is the negative
# prompt (`pipe.default_negative_prompt_image` drops temporal/motion terms). Uses the same
# disk-offload VRAM profile as the low-VRAM t2v/ti2v examples.
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
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

with open(os.path.join(os.path.dirname(__file__), "..", "model_inference", "prompts", "t2i_example.json"), "r", encoding="utf-8") as f:
    caption = json.load(f)

frames = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt_image,
    height=480, width=832, num_frames=1,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
frames[0].save("image_lingbot-video-dense-1.3b_t2i_low_vram.png")
