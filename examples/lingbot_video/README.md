# LingBot-Video

[LingBot-Video](https://github.com/modelscope) is a flow-matching video-generation model. This directory provides DiffSynth-Studio inference and training (LoRA SFT) support for the **Dense-1.3B** text-to-video checkpoint.

The integration is built on the standard DiffSynth pipeline stack:

- **DiT** — `LingBotVideoDiT` (`diffsynth/models/lingbot_video_dit.py`), the video denoiser. The Dense-1.3B build uses a plain FFN; the architecture also supports an MoE FFN.
- **Text encoder** — `LingBotVideoTextEncoder` (Qwen3-VL). Prompts are wrapped in a prompt-enhancement chat template, encoded, and the template-prefix tokens are cropped.
- **VAE** — reuses DiffSynth's `QwenImageVAE` (byte-identical to the LingBot-Video VAE), 8× spatial / 4× temporal.
- **Scheduler** — DiffSynth's `FlowMatchScheduler` (Wan template): first-order flow-matching Euler for inference; training uses the full-resolution 1000-step flow-matching schedule.

## Installation

Follow the top-level DiffSynth-Studio installation. LingBot-Video additionally requires `transformers >= 5.x` (for Qwen3-VL) and `imageio` / `imageio-ffmpeg` for video I/O.

```bash
pip install -e .
```

## Model download

```bash
modelscope download --model Robbyant/lingbot-video-dense-1.3b --local_dir ./models/Robbyant/lingbot-video-dense-1.3b
```

Both the inference and training examples use `model_id`-based configs, which download the required files automatically the first time they run, so the manual download above is optional. You can also point `ModelConfig(path=...)` at local files if you already have them.

## Inference

```bash
python examples/lingbot_video/model_inference/lingbot-video-dense-1.3b.py
```

Minimal text-to-video:

```python
import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig

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
video = pipe(prompt="A playful puppy runs across a lush green meadow ...", height=480, width=832, num_frames=81, seed=0)
save_video(video, "output.mp4", fps=15, quality=5)
```

The pipeline ships a default (T2V) negative prompt, so `negative_prompt` is optional. Video-to-video is supported by passing `input_video=` (a list of frames or a `VideoData`) together with `denoising_strength < 1`.

**Low VRAM:** pass `vram_limit=<GB>` to `from_pretrained` to enable layer-by-layer offloading — see `model_inference_low_vram/lingbot-video-dense-1.3b.py`.

### Image-to-video (TI2V)

Condition on a **first frame** by passing a `PIL.Image` as `input_image`; the model animates it. Dense-1.3B reuses the same T2V checkpoint (no separate i2v weight). The frame is used twice — as a visual reference for the Qwen3-VL text encoder, and as a clean latent pinned into the first frame of the diffusion latent before sampling and after every step, so only the following frames are generated.

```bash
python examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_ti2v.py
```

```python
from PIL import Image
video = pipe(
    prompt=caption,                                  # describes the motion from the first frame
    input_image=Image.open("first_frame.png").convert("RGB"),
    height=480, width=832, num_frames=81, cfg_scale=3.0, seed=0,
)
```

The runnable example uses the released first frame + caption in `model_inference/assets/ti2v_first_frame.png` and `model_inference/prompts/ti2v_example.json`. `input_image` and `input_video` are mutually exclusive.

## Prompt rewriting (important for quality)

LingBot-Video is trained on **structured-JSON captions**, not free-form prose. Feeding a flat sentence is out-of-distribution and visibly degrades quality (softer, less coherent motion); feeding the structured caption the model expects restores it. The pipeline accepts a caption as a `dict`, a path to a `prompt.json`, or a plain string, and normalises it to the exact compact-JSON format the DiT was trained on — a plain string is passed through unchanged, so existing scripts keep working.

```python
# All three are equivalent once normalised, and all are accepted by pipe(prompt=...):
pipe(prompt={"caption": {"comprehensive_description": {...}}, "duration": 5})   # dict
pipe(prompt="assets/cases/t2v/example_1/prompt.json")                          # prompt.json path
pipe(prompt='{"comprehensive_description":{...}}')                             # already-serialised string
```

To turn a **brief idea** into that structured caption, use the two-stage rewriter shipped here (`model_inference/prompt_rewriter.py`), a faithful port of the original: stage 1 *expands* the idea into a natural-language caption, stage 2 *maps* it into structured JSON.

```python
# The rewriter lives in model_inference/; run from that directory (or add it to
# sys.path) so this sibling import resolves.
from prompt_rewriter import rewrite_prompt
caption = rewrite_prompt("a puppy running across a meadow", mode="t2v", duration=5)
video = pipe(prompt=caption, height=480, width=832, num_frames=81, cfg_scale=3.0)
```

The rewriter is a **separate VLM + stage-2 LoRA adapter** (not the DiT). Point it at the weights via `REWRITER_BASE_MODEL` / `REWRITER_ADAPTER` (or `base=`/`adapter=`). If you serve the rewriter behind a hosted / OpenAI-compatible endpoint instead of loading it locally, pass a custom object exposing `generate(text, image, use_lora)` as `backend=`. See the optional rewrite section at the bottom of `model_inference/lingbot-video-dense-1.3b.py`.

## Training (LoRA SFT)

`model_training/train.py` fine-tunes the DiT with LoRA using the flow-matching SFT objective.

```bash
bash examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b.sh
```

The launch script first downloads the example video-SFT dataset used across DiffSynth-Studio, then trains on it:

```bash
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset \
    --include "wanvideo/Wan2.1-T2V-1.3B/*" --local_dir ./data/diffsynth_example_dataset
```

### Dataset format

Bring your own data by pointing `--dataset_base_path` / `--dataset_metadata_path` at a metadata CSV (or JSONL) with a `video` column (path relative to `--dataset_base_path`) and a `prompt` column — the same layout as the example dataset above:

```
video,prompt
videos/000.mp4,A playful puppy runs across a lush green meadow ...
videos/001.mp4,A serene lake at sunrise, mist rising from the water ...
```

Pass `--data_file_keys "video"` so the loader treats the `video` column as a file to load.

For best results the `prompt` column should hold **structured-JSON captions** (the same in-distribution format used at inference — see [Prompt rewriting](#prompt-rewriting-important-for-quality)). `train.py` runs each prompt through `normalize_caption`, so a `dict`-valued prompt (in JSONL) or a path to a `prompt.json` is serialised automatically, and a plain string is used as-is. If your dataset stores raw prose, rewrite it once offline before training:

```bash
python examples/lingbot_video/model_training/rewrite_captions.py \
    --metadata metadata.csv --output metadata_rewritten.csv \
    --base /path/to/rewriter-base --adapter /path/to/rewriter-step2-lora --duration 5
```

then train on `metadata_rewritten.csv`. (This is done offline because running the rewriter VLM inside the dataloader on every step would be prohibitively slow.)

### Attention-only LoRA (default scope)

The launch script patches LoRA on the joint text+video self-attention only:

```
--lora_base_model "dit"
--lora_target_modules "to_q,to_k,to_v,to_out"
--lora_rank 32
--remove_prefix_in_ckpt "pipe.dit."
```

The MoE / FFN experts (`gate_proj`, `up_proj`, `down_proj`) and the router are left frozen. To also adapt the FFN, add those module names to `--lora_target_modules`.

### Useful flags

- `--use_gradient_checkpointing` — trade compute for memory (recommended; the trainer enables it regardless).
- `--num_frames`, `--height`, `--width` — training clip shape (`num_frames` must satisfy `4k+1`; H/W divisible by 16).
- `--max_timestep_boundary` / `--min_timestep_boundary` — restrict the sampled training timesteps to a sub-range of the schedule.
- `--lora_checkpoint <path>` — resume / continue from a previously trained LoRA.

### Applying a trained LoRA

Trained LoRA checkpoints are written to `--output_path` with the `pipe.dit.` prefix stripped (keys like `blocks.0.attn.to_q.lora_A.weight`). To continue training from one, pass it via `--lora_checkpoint`.

## Notes

- The text encoder shares its checkpoint fingerprint with the existing `krea2_text_encoder` (identical Qwen3-VL architecture), so the model loader instantiates both when loading LingBot-Video. This is redundant load time only — the pipeline fetches the correct encoder by name and the other is released.
- The 5D-video VAE encode/decode and its latent normalisation live in the pipeline (`LingBotVideoPipeline.encode_video` / `decode_video`), so `QwenImageVAE` stays byte-identical to its image use elsewhere; the pipeline does not separately re-apply `latents_mean` / `latents_std`.
