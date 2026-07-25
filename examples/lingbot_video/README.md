# LingBot-Video

[LingBot-Video](https://github.com/modelscope) is a flow-matching video-generation model. This directory provides DiffSynth-Studio inference and training (LoRA SFT) support for the **Dense-1.3B** text-to-video checkpoint.

The integration is built on the standard DiffSynth pipeline stack:

- **DiT** — `LingBotVideoDiT` (`diffsynth/models/lingbot_video_dit.py`), the video denoiser. The Dense-1.3B build uses a plain FFN; the architecture also supports an MoE FFN.
- **Text encoder** — `LingBotVideoTextEncoder` (Qwen3-VL). Prompts are wrapped in a prompt-enhancement chat template, encoded, and the template-prefix tokens are cropped.
- **VAE** — reuses DiffSynth's `QwenImageVAE` (byte-identical to the LingBot-Video VAE), 8× spatial / 4× temporal.
- **Scheduler** — `LingBotVideoUniPCScheduler`: UniPC multistep for inference; it falls back to the full-resolution flow-matching schedule for training.

## Installation

Follow the top-level DiffSynth-Studio installation. LingBot-Video additionally requires `transformers >= 5.x` (for Qwen3-VL) and `imageio` / `imageio-ffmpeg` for video I/O.

```bash
pip install -e .
```

## Model download

```bash
modelscope download --model Robbyant/lingbot-video-dense-1.3b --local_dir ./models/Robbyant/lingbot-video-dense-1.3b
```

The inference examples below use `ModelConfig(model_id=...)`, which downloads the required files automatically the first time they run. You can also point `ModelConfig(path=...)` at local files (see the training script).

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

**Low VRAM:** pass `vram_limit=<GB>` to `from_pretrained` to enable layer-by-layer offloading — see `model_inference/lingbot-video-dense-1.3b_low_vram.py`.

## Training (LoRA SFT)

`model_training/train.py` fine-tunes the DiT with LoRA using the flow-matching SFT objective.

```bash
bash examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b.sh
```

### Dataset format

A metadata CSV (or JSONL) with a `video` column (path relative to `--dataset_base_path`) and a `prompt` column:

```
video,prompt
videos/000.mp4,A playful puppy runs across a lush green meadow ...
videos/001.mp4,A serene lake at sunrise, mist rising from the water ...
```

Pass `--data_file_keys "video"` so the loader treats the `video` column as a file to load.

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
- Latent normalisation is handled inside the VAE's 5D-video code path; the pipeline does not re-apply `latents_mean` / `latents_std`.
