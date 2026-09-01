# LTX-2.5

DiffSynth-Studio provides portable LTX-2.5 joint audio-video inference through
`LTX25AudioVideoPipeline`. It loads the official split BF16 checkpoints locally
and does not require `ltx_core`, NATTEN, Triton, or ltx-kernels at runtime.

> LTX-2.5 weights are gated. Obtain access from Lightricks and place the
> checkpoint files in a local model directory before running an example.

## Implemented components

- LTX-2.5 22B Distilled and Dev DiT checkpoints
- Fine-tuned Gemma4 12B encoder, packed tokenizer assets, and dual AV connectors
- Duration head and automatic causal-grid frame-count prediction
- DiffVAE video encoder and pure-PyTorch eager diffusion decoder
- Audio VAE, 48 kHz BWE vocoder, spatial x2 latent upsampler, and temporal x2
  upsampler registration
- Dev stage-2 distilled-LoRA loading

The eager DiffVAE decoder uses tiled scaled-dot-product attention as a portable
neighborhood-attention fallback. Its fixed-input output matches the upstream
eager implementation for deterministic decoder stages and an x0 diffusion step.

## Inference modes

The public `LTX25AudioVideoPipeline` API follows the existing LTX-2.3 API.

| Mode | Pipeline parameters | Status |
|---|---|---|
| Distilled two-stage T2AV | `use_distilled_pipeline=True`, `use_two_stage_pipeline=True` | Supported |
| Distilled two-stage I2AV | `input_images`, `input_images_indexes` | Supported |
| Dev one-stage T2AV | `use_distilled_pipeline=False`, `use_two_stage_pipeline=False` | Supported |
| Dev one-stage I2AV | `input_images`, `use_two_stage_pipeline=False` | Supported |
| Dev two-stage T2AV/I2AV | `stage2_lora_config`, `use_two_stage_pipeline=True` | Supported |
| Audio-to-video | `retake_audio`, `audio_sample_rate`, optional `retake_audio_regions` | Supported |
| Video/audio retake | `retake_video`, `retake_video_regions`, `retake_audio_regions` | Supported |
| Keyframe interpolation | multiple `input_images` and `input_images_indexes` | Supported |
| Pixel Spatial Upscaler IC-LoRA | `in_context_videos`, `in_context_downsample_factor=2` | Supported |

For a two-stage Dev run, supply the released distilled stage-2 LoRA through
`stage2_lora_config`. The two-stage path is required for distilled inference;
Dev also supports a one-stage path without a stage-2 LoRA.

The Pixel Spatial Upscaler requires the official LTX-2.5 Pixel IC-LoRA.
Load it with `pipe.load_lora(pipe.dit, ModelConfig(path=...))`, pass the
reference video through `in_context_videos`, and set
`clear_lora_before_state_two=True`. Its reference resolution is one quarter of
the final height and width: stage 1 is half resolution and the adapter's
`reference_downscale_factor` is 2. Do not load an LTX-2.3 adapter into an
LTX-2.5 DiT.

## Geometry and memory requirements

- `num_frames % 8 == 1`
- One-stage height and width must be divisible by 32.
- Two-stage height and width must be divisible by 64.
- The low-VRAM examples use BF16 compute, FP8 CPU weight offload, and
  fine-grained management for the DiT, Gemma4 encoder, text connectors, and
  DiffVAE decoder.
- `LTX25_VRAM_LIMIT_GB` controls the GPU budget used to retain prepared model
  layers. It defaults to 16; it is not a hard end-to-end VRAM limit.
- A 960×576×121 distilled T2AV run with `LTX25_VRAM_LIMIT_GB=16` measured a
  31.8 GiB PyTorch allocation peak and 44.4 GiB reserved peak. Treat 48 GiB
  as a measured lower bound for this resolution, not a validated 48 GiB target,
  and leave headroom for the driver.
- `tiled=True` does not yet split the portable DiffVAE decode volume. The
  decoder determines the current full-resolution peak; lower-card support
  requires a tiled decoder implementation.

Run the low-VRAM examples with:

```bash
python examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-DistilledPipeline.py
python examples/ltx2/model_inference_low_vram/LTX-2.5-Keyframe-Interpolation.py
python examples/ltx2/model_inference_low_vram/LTX-2.5-IC-LoRA-Pixel-Spatial-Upscaler.py
```

The split checkpoints are expected under `models/Lightricks/LTX-2.5`. The
Pixel example additionally expects the separately gated adapter under
`models/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`. Set a larger
prepared-layer budget with `LTX25_VRAM_LIMIT_GB=24` when GPU memory is
available; it can improve speed but does not lower the decoder peak. Adjust
paths in an example if the local model directory is different.

## Scope

This integration is inference-only. Training, LTX-2.5-specific DFR, Dub-It,
and HDR/EXR pipelines are not included in this scope.
