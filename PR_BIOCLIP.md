# Add BioCLIP-2 Metric

## Summary

Adds a new `BioCLIPMetric` to `diffsynth.metrics` for biological image–text similarity scoring, based on the [BioCLIP-2](https://huggingface.co/imageomics/bioclip-2) model (ViT-L/14 trained on TreeOfLife-200M).

This integration follows the same pattern as the existing `CLIPMetric`, returning `cosine_similarity * logit_scale.exp()` so the output format is consistent with other CLIP-family metrics already in the repo.

## Files

**New:**
- `diffsynth/models/bioclip.py` — `BioCLIPv2Model` (HF `CLIPModel` subclass with ViT-L/14 config) and `BioCLIPv2Compute` (preprocessing + inference wrapper, mirrors the `CLIPModel` wrapper pattern).
- `diffsynth/metrics/bioclip.py` — `BioCLIPMetric` with `from_pretrained` / `compute(prompt, images)` API.
- `examples/image_quality_metric/bioclip.py` — Usage example.

**Modified:**
- `diffsynth/configs/model_configs.py` — Registers BioCLIPv2 in `image_metrics_series` (model_hash `3a020a3e47afb7c5e21c52f2d0692c09`, reuses the existing `ImageMetricsOpenCLIPStateDictConverter`).
- `diffsynth/metrics/__init__.py` — Exports `BioCLIPMetric`.

## Design notes

- **No `open_clip` dependency.** BioCLIP-2 ships in OpenCLIP format. Instead of pulling in `open_clip`, we reuse the existing `ImageMetricsOpenCLIPStateDictConverter` to remap keys to HuggingFace `CLIPModel` layout, which is already a transitive dependency of DiffSynth.
- **Preprocessing.** Standard CLIP pipeline read from `open_clip_config.json`: resize shortest side to 224 (bicubic) → center crop 224×224 → ToTensor → ImageNet-style normalize with CLIP mean/std.
- **Tokenizer.** Loaded via `transformers.CLIPTokenizer.from_pretrained` from the BioCLIPv2 directory's HF-format tokenizer files (vocab.json, merges.txt, tokenizer.json), so no extra tokenizer code is needed.
- **Weights.** Uploaded to ModelScope at `DiffSynth-Studio/ImageMetrics/BioCLIPv2/` (uses the original OpenCLIP-format `open_clip_model.safetensors` + tokenizer files).

## Verification

Bit-exact consistency with the official `open_clip` library on the same input:

| Quantity | Max abs diff |
|---|---|
| Image features (normalized) | 1.34e-7 |
| Text features (normalized) | 1.12e-7 |
| Cosine similarity | 0.0 |

Tested with `ViT-L-14` architecture loaded from the same `open_clip_model.safetensors` file, comparing `encode_image` / `encode_text` outputs against `BioCLIPv2Compute.get_image_features` / `get_text_features`. Differences are within float32 numerical precision.

## Usage

```python
from diffsynth.metrics import BioCLIPMetric, ModelConfig
from PIL import Image

metric = BioCLIPMetric.from_pretrained(
    model_config=ModelConfig(
        model_id="DiffSynth-Studio/ImageMetrics",
        origin_file_pattern="BioCLIPv2/open_clip_model.safetensors",
    ),
    device="cuda",
)
score = metric.compute("a photo of a bird", Image.open("img.jpg"))[0]
```

## Test plan

- [x] State dict loads with 0 missing / 0 unexpected keys
- [x] Output matches `open_clip` reference within float32 precision
- [x] Example script runs end-to-end on GPU
- [x] Model hash matches DiffSynth's `hash_model_file` output
