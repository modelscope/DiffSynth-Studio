# Add LPIPS image-quality metric

## Summary

Adds **LPIPS** (Learned Perceptual Image Patch Similarity, [Zhang et al. CVPR 2018](https://arxiv.org/abs/1801.03924)) to `diffsynth.metrics`, alongside the existing FID / CLIP / Aesthetic / PickScore / ImageReward / HPSv2 / HPSv3 metrics. Reference implementation: [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity).

Three backbone variants (`alex` / `vgg` / `squeeze`) are supported and selectable through a single `net=...` flag — the matching `safetensors` weight file is auto-resolved when no `model_config` is given.

## Files

### New

| File | Purpose |
|------|---------|
| `diffsynth/models/lpips.py` | Self-contained backbones (AlexNet / VGG16 / SqueezeNet1.1 features), `ScalingLayer`, `NetLinLayer`, top-level `LPIPSModel`, and `LPIPSCompute` (handles file/dir input, stem matching, conditional resize). No `torchvision.models` weight fetch — the registered safetensors carry every parameter. |
| `diffsynth/metrics/lpips.py` | `LPIPSMetric.from_pretrained(net, ...)` matching the existing `FIDMetric` shape. Auto-derives the `ModelConfig` and `model_pool.fetch_model(...)` name from `net`. |
| `examples/image_quality_metric/lpips.py` | Example covering both `img-vs-img` and `dir-vs-dir` calls on the existing FLUX example dataset. |

### Modified

| File | Change |
|------|--------|
| `diffsynth/metrics/__init__.py` | Export `LPIPSMetric` |
| `diffsynth/configs/model_configs.py` | Three new entries in `image_metrics_series` (one per backbone), each with `extra_kwargs={"net": ...}` |
| `diffsynth/utils/state_dict_converters/image_metrics.py` | Add `ImageMetricsLPIPSStateDictConverter` (identity converter — the uploaded safetensors already match `LPIPSModel.state_dict()`) |

No other files changed; conda environment, other metrics, README, and docs are untouched.

## Public API

```python
from diffsynth.metrics import LPIPSMetric

# Default: alex backbone, file = LPIPS/alexnet.safetensors (~9.9 MB)
metric = LPIPSMetric.from_pretrained(net="alex", device="cuda")

# img vs img -> single float
score = metric.compute("a.png", "b.png")

# dir vs dir -> mean over filename-stem-matched pairs (float)
score = metric.compute("./dir_a", "./dir_b")
```

Other supported kwargs: `net="vgg"|"squeeze"`, `target_size=512`, `batch_size=16`, `num_workers=0`, plus an optional explicit `model_config=ModelConfig(...)` to override the default weight file.

## Behavior

**`compute(image_a, image_b)`** dispatches by input type:

| Both inputs | Behavior |
|-------------|----------|
| Image files / `PIL.Image` | If sizes match → no resize. If sizes differ → `Resize(target_size, BICUBIC)` + `CenterCrop(target_size)` (consistent with `diffsynth.models.image_reward`'s pattern). Returns a single float. |
| Directories | Pair by filename stem (e.g. `dog.png` ↔ `dog.jpg` match; orphan files are ignored). If **all** images across both dirs share the same `(H, W)` → no resize; otherwise resize all. Returns the mean LPIPS over matched pairs. |
| Mixed (one file, one dir) | `ValueError` |

After `ToTensor`, values are clamped to `[0, 1]` before being mapped to the official `[-1, 1]` LPIPS input range — this guards against BICUBIC overshoot (other metrics in this repo also use BICUBIC; FID and ImageReward do not clamp, but LPIPS is sensitive to out-of-range inputs because `ScalingLayer` applies a per-channel mean/std).

## Weights (uploaded to ModelScope)

The three weight files are committed under `DiffSynth-Studio/ImageMetrics/LPIPS/` on ModelScope. Each one is a complete LPIPS state dict — `net.slice{1..N}.*` (backbone), `scaling_layer.shift/scale` (ImageNet color buffers), and 5 or 7 `lin{i}.model.1.weight` 1×1 conv weights — produced by combining the official torchvision ImageNet checkpoints with the LPIPS lin-layer weights from `richzhang/PerceptualSimilarity`'s `lpips/weights/v0.1/`.

| File | Size | Hash (md5) | `model_name` |
|------|------|------------|--------------|
| `LPIPS/alexnet.safetensors` | ~9.9 MB | `08a75c660c9b2e775c530a0955857f1f` | `image_metrics_lpips_alex` |
| `LPIPS/vgg.safetensors` | ~58.9 MB | `5740953aaa8aba2ecd9b9c23da813591` | `image_metrics_lpips_vgg` |
| `LPIPS/squeezenet.safetensors` | ~2.9 MB | `ff994b70a30599287a332105396d5004` | `image_metrics_lpips_squeeze` |

## Consistency with existing metrics

- `LPIPSMetric` subclasses the same `Metric` base used by every other metric, and uses the standard `download_and_load_models` → `model_pool.fetch_model(...)` flow.
- `from_pretrained(...)` follows the FID / CLIP signature shape: optional `model_config`, `device`, `vram_limit`, plus metric-specific kwargs.
- All three backbones are registered in `image_metrics_series` with the same shape as the FID entry, just differentiated by `extra_kwargs={"net": ...}`.
- The example file mirrors `examples/image_quality_metric/fid.py` (download via `dataset_snapshot_download`, then `metric.compute(...)`).

## Test plan

Tests run inside the user-provided `compound` conda env on CPU (login node had no GPU); the code path is device-agnostic.

- [x] Numerical parity vs official `lpips` package on `PerceptualSimilarity/imgs/ex_dir{0,1}` (64×64, no resize):

  | net | DiffSynth (mean) | Official `lpips` | abs diff |
  |-----|------------------|-------------------|----------|
  | alex | 0.429723 | 0.429723 | 6.7e-08 |
  | vgg  | 0.495139 | 0.495139 | 1.5e-08 |
  | squeeze | 0.429475 | 0.429475 | 6.0e-08 |

  Per-pair img-vs-img scores match to `0.000000` for all 6 (3 nets × 2 pairs).

- [x] State dict cross-check: every common key between the new safetensors and `lpips.LPIPS(net=...).state_dict()` is `torch.equal`-identical (alex 17/17, vgg 33/33, squeeze 59/59 keys; the only `lins.*` keys missing are `nn.ModuleList` aliases that point at the same tensors).

- [x] `LPIPSModel.load_state_dict(...)` reports `0` missing and `0` unexpected keys for all three weight files.

- [x] `model_pool.auto_load_model(...)` correctly identifies and loads the right backbone by hash for all three files.

- [x] Behavioral edge cases:
  - Same image vs itself → `0.0` (alex, exact)
  - Different-sized images → BICUBIC resize path runs, returns a sensible non-zero score
  - Mixed-size directory pair → all images are resized, returns mean
  - Stem matching `dog.png` ↔ `dog.jpg` works
  - Mixed input (one file, one directory) → `ValueError`

- [x] Example script `examples/image_quality_metric/lpips.py` runs end-to-end (`alex` backbone, FLUX dataset). The `dir-vs-dir` score is `0.0000` because the `flux/FLUX.1-dev` and `flux2/FLUX.2-dev` example dirs contain byte-identical images (same as the FID example exhibits very-near-zero behavior); the `img-vs-img` call between two distinct images returns a sensible non-zero score.

## Out of scope

- README / `docs/.../Image-Quality-Metrics.md` table updates — left for a docs-only follow-up.
- LPIPS as a training loss — only the inference metric path is added.
- Resize strategies beyond center-crop + 512×512 BICUBIC — a single `target_size` knob covers the use cases requested.
