import os
import torch
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig, normalize_caption


# LingBot-Video is trained on STRUCTURED-JSON captions, not free-form prose. Feeding a
# flat sentence is out-of-distribution and visibly degrades quality; feeding the
# structured caption the model expects restores it. This example runs on a released
# in-distribution caption by default, and shows at the bottom how to turn a brief idea
# into such a caption with the two-stage prompt rewriter.

pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="processor/"),
)

# --- Text-to-video -------------------------------------------------------------------
# The prompts/t2v_example_*.json files are released LingBot-Video t2v captions — real
# in-distribution examples you can copy as templates for your own. normalize_caption
# accepts a dict, a list, or a path to a prompt.json; the pipeline calls it internally
# too, so pipe(prompt="prompts/t2v_example_1.json") works just as well. The default
# (T2V) negative prompt is built into the pipeline, so negative_prompt can be left unset.
caption = normalize_caption(os.path.join(os.path.dirname(__file__), "prompts", "t2v_example_1.json"))
video = pipe(
    prompt=caption,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-dense-1.3b.mp4", fps=15, quality=10)

# --- Video-to-video ------------------------------------------------------------------
# denoising_strength < 1 keeps part of the input structure.
input_video = VideoData("video_lingbot-video-dense-1.3b.mp4", height=480, width=832)
video = pipe(
    prompt=caption,
    input_video=input_video, denoising_strength=0.7,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=1,
)
save_video(video, "video_lingbot-video-dense-1.3b_v2v.mp4", fps=15, quality=10)

# --- Optional: rewrite a brief idea into a structured caption ------------------------
# If you only have a brief idea (or free-form prose), turn it into the structured caption
# the model expects with the two-stage rewriter in prompt_rewriter.py (a sibling module
# here). The rewriter is a separate VLM + stage-2 LoRA adapter (NOT the DiT), so it is
# NOT downloaded automatically — fetch both weights first, then point the env vars at them:
#     modelscope download --model Qwen/Qwen3.6-27B --local_dir ./models/Qwen/Qwen3.6-27B
#     modelscope download --model Robbyant/lingbot-video-rewriter-lora --local_dir ./models/Robbyant/lingbot-video-rewriter-lora
#     export REWRITER_BASE_MODEL=./models/Qwen/Qwen3.6-27B
#     export REWRITER_ADAPTER=./models/Robbyant/lingbot-video-rewriter-lora
# To drive a hosted / OpenAI-compatible endpoint instead of a local VLM, pass a custom
# object exposing generate(text, image, use_lora) as backend=.
#
# from prompt_rewriter import rewrite_prompt
# caption = rewrite_prompt(
#     "A playful puppy runs across a lush green meadow, chasing a red ball. "
#     "Dynamic side-tracking camera.",
#     mode="t2v", duration=5,
# )
# video = pipe(prompt=caption, height=480, width=832, num_frames=81, cfg_scale=3.0, seed=0)
# save_video(video, "video_lingbot-video-dense-1.3b_rewrite.mp4", fps=15, quality=10)
