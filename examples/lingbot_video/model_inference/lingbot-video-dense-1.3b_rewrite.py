import os
import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig
from diffsynth.pipelines.lingbot_video_prompt_rewriter import rewrite_prompt, normalize_caption


# LingBot-Video is trained on STRUCTURED-JSON captions, not free-form prose. Feeding a
# flat sentence is out-of-distribution and visibly degrades quality; rewriting the idea
# into the structured caption the model expects restores it. This example shows the two
# supported ways to obtain that caption.

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


# --- Option A: rewrite a brief idea with the two-stage prompt rewriter --------------
# The rewriter is a separate VLM + LoRA adapter (NOT the DiT). Point these at the
# rewriter base model and its stage-2 adapter:
#     export REWRITER_BASE_MODEL=/path/to/rewriter-base
#     export REWRITER_ADAPTER=/path/to/rewriter-step2-lora
# If you serve the rewriter behind a hosted / OpenAI-compatible endpoint instead, pass
# a custom backend object exposing generate(text, image, use_lora) as `backend=`.
if os.environ.get("REWRITER_BASE_MODEL") and os.environ.get("REWRITER_ADAPTER"):
    caption = rewrite_prompt(
        "A playful puppy runs across a lush green meadow, chasing a red ball. "
        "Dynamic side-tracking camera.",
        mode="t2v", duration=5, backend="transformers",
    )
    print("Rewritten caption:\n", caption)
else:
    # --- Option B: no rewriter model — supply a structured caption directly ----------
    # A dict (or a path to a prompt.json) is serialised to the exact model format by
    # normalize_caption; the pipeline also calls it internally, so pipe(prompt=<dict>)
    # works too. Replace this stub with a real structured caption for best quality.
    caption = normalize_caption({
        "caption": {
            "comprehensive_description": {
                "scene_content_description": "A playful golden puppy runs across a lush green meadow dotted with wildflowers, chasing a bright red ball under a clear blue sky.",
                "camera_movement_description": "Dynamic side-tracking shot following the puppy.",
            }
        },
        "duration": 5,
    })

video = pipe(
    prompt=caption,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video_lingbot-video-dense-1.3b_rewrite.mp4", fps=15, quality=10)
