import json
import torch
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

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="lingbot_video/lingbot-video-dense-1.3b_t2i/*",
)
with open("data/diffsynth_example_dataset/lingbot_video/lingbot-video-dense-1.3b_t2i/t2i_example.json", "r", encoding="utf-8") as f:
    caption = json.load(f)

frames = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt_image,
    height=480, width=832, num_frames=1,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
frames[0].save("image_lingbot-video-dense-1.3b_t2i.png")
