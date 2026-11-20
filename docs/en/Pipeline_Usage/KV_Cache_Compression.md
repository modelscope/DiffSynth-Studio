# KV Cache Compression

`QwenImage21Pipeline` caches the conditioning prefix (the prompt and reference-image tokens) as a
KV cache during the first denoising step and reuses it for every later step. The cache is often the
largest live tensor group in the pipeline: with eight 1024×1024 reference images the prefix is
32 510 tokens per transformer block, which is 16.3 GiB per CFG branch and 32.5 GiB for the positive
and negative caches together — more than the 30 GiB of model weights.

`kv_compress_config` stores that cache compressed. The cache is compressed once, when it is
produced, and decompressed temporarily whenever a later denoising step reads it.

## Quick start

```python
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
import torch
from PIL import Image

pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
)

image = pipe(
    "Generate a group photo of these characters.",
    edit_image=[Image.open(path) for path in image_paths],
    seed=1,
    cfg_scale=4,
    kv_compress_config=[{"method": "entropack", "target_bpp": 6.0}],
)
image.save("image.png")
print(pipe.last_kv_compression_stats)
```

`kv_compress_config` is a **list of compression stages**, applied in order (a single dict is
accepted as a one-stage list); `None` (the default) keeps the uncompressed cache.
`template_kv_cache` and `negative_template_kv_cache` are a different mechanism and are not affected.

## Compression stages

| stage | Config example | What it stores |
| --- | --- | --- |
| `svd` | `{"method": "svd", "rank": 64}` | low-rank factors `U`, `S`, `V` via `torch.svd_lowrank` |
| `cholesky_qr` | `{"method": "cholesky_qr", "rank": 1024}` | the same low-rank factors, decoded with a Cholesky-QR range finder |
| `entropack` | `{"method": "entropack", "target_bpp": 4.0}` | EntroPack entropy-coded values |

Stages compose by listing them: `[{"method": "svd", "rank": 1024}, {"method": "entropack",
"target_bpp": 8.0}]` stores SVD factors that are entropy coded in turn, and decompression runs the
stages in reverse. A future stage only needs to subclass `Scheme` and call `register_scheme`.

### SVD

The prefix tensors are viewed as `batch × heads` matrices of shape
`[prefix_tokens, head_dim]` and decomposed with a fast randomized SVD. `rank` selects the number of
singular triplets, `rank_ratio` selects it as a fraction of `min(m, n)`, and `energy` keeps the
smallest rank reaching a captured-energy fraction.

`engine` chooses the implementation: `"svd_lowrank"` calls `torch.svd_lowrank` (PyTorch's randomized
approximate SVD), `"randomized"` applies the same randomized range finder with batched matmuls and
Cholesky-QR (much faster on accelerators with slow batched QR kernels), and `"auto"` uses
`"randomized"` on CUDA.

### Cholesky-QR (`cholesky_qr`)

The same randomized SVD as `svd`, but the range finder is orthonormalized with two Cholesky-QR
passes instead of `torch.linalg.qr`. Identical result and compression ratio, 4–8× faster encoding
when batched QR is slow (head rank=32: 0.07 s vs 0.55 s per KV tensor; joint rank=1024: 0.23 s vs
0.85 s). Options are the same as `svd`.

### EntroPack

[EntroPack](https://github.com/modelscope/entropack) provides GPU entropy coding. `scheme="lossy"`
achieves a target `target_bpp` (0.001–11 bits per element), `scheme="tile_ans"` and
`scheme="dfloat11"` are lossless. It needs `entropack` installed together with a working CuPy build;
without a CUDA backend the pure-PyTorch fallback is orders of magnitude slower:

```bash
pip install "entropack[cuda13]"   # cupy-cuda13x
# or, when the GPU vendor ships its own CuPy build, install that instead
```

EntroPack's lossy scheme allocates a quantization scale per row, so narrow rows lose precision. The
codec groups consecutive rows until each row holds `row_width` values (128 by default), and stores
tensors below `min_numel` (4096 by default) without entropy coding.

### SVD + EntroPack

The stacked mode applies EntroPack to the SVD factors, multiplying the savings of both stages.
Note that it inherits the SVD's truncation error: for a given VRAM budget EntroPack alone usually
keeps more information (see the measurements below).


## Per-head vs joint SVD (`layout`)

The KV tensor has shape `[batch, seq, heads, head_dim]`. `layout` decides which matrix the SVD
is taken on:

* `"head"` (default): one `[seq, head_dim]` matrix per `(batch, head)` pair — the SVD runs
  between the sequence axis and the head-dim axis.
* `"joint"`: one `[seq, heads * head_dim]` matrix per batch item — the SVD runs between the
  sequence axis and the flattened head/dim axes.

The joint layout is much more efficient for this cache because the joint spectrum decays far
faster: at ~25% of the dimensions (rank 1024 of 4096) it keeps 98.7% of the energy, while the
per-head layout at 25% (rank 32 of 128) keeps only 84%. Measured on real prefix tensors:

| config | KV error (rel L2) | ratio |
| --- | --- | --- |
| per-head SVD rank=32 | 0.399 | 3.98× |
| joint SVD rank=1024 | **0.112** | 3.55× |
| per-head SVD rank=64 | 0.237 | 1.99× |
| joint SVD rank=2048 | **0.049** | 1.78× |

Generation quality (8 images, 1024², 40 steps):

| config | peak VRAM | total time | ratio | PSNR | SSIM | result |
| --- | --- | --- | --- | --- | --- | --- |
| joint rank=2048 | 53.4 GiB | 151.5 s | 1.78× | 16.3 dB | 0.727 | nearly identical to baseline |
| joint rank=1024 | 44.5 GiB | 92.3 s | 3.55× | 13.2 dB | 0.594 | coherent, identities drift |
| joint rank=512 | 40.0 GiB | 73.3 s | 7.11× | 11.3 dB | 0.517 | coherent but waxy artifacts |
| per-head rank=64 | 51.4 GiB | 85.0 s | 1.99× | 9.0 dB | 0.282 | destroyed |
| per-head rank=32 | 43.5 GiB | 76.9 s | 3.98× | 8.7 dB | 0.303 | destroyed |

So the joint layout makes low-rank KV compression usable, but it is still dominated by EntroPack
at the same VRAM budget (EntroPack 4 bpp: 3.98× with 0.081 error, 5.6 s to encode vs 71 s).
The joint SVD pays for its encoder: a rank-1024 decomposition costs ~27 s for the 8-image cache.

## Statistics

`pipe.last_kv_compression_stats` returns a dictionary describing the cache of the last call:

```json
{"method": "entropack(lossy@6.0bpp)", "compress_calls": 128, "decompress_calls": 4992,
 "original_gib": 31.746, "stored_gib": 11.984, "compression_ratio": 2.649,
 "effective_bpp": 6.04, "compress_seconds": 5.45, "decompress_seconds": 2.67}
```

`compress_seconds` is the one-off encoding cost of the first denoising step; `decompress_seconds`
sums the temporary decodes of all later steps.

## Generic tensor API

The same codecs compress any tensor, not only the KV cache:

```python
from diffsynth.core.context_compress import compress_tensor, decompress_tensor

store = compress_tensor(tensor, [
    {"method": "svd", "rank": 32},
    {"method": "entropack", "target_bpp": 8.0},
])
print(store.nbytes, store.compression_ratio, store.bits_per_element)
dense = decompress_tensor(store)
store = store.to("cpu")   # move the stored state without decoding it
```

The code lives in `diffsynth/core/context_compress`.

## Measurements

Eight reference images, 1024×1024 output, 40 steps, `cfg_scale=4`, one PPU accelerator
(96 GiB). Quality is measured against the uncompressed result of the same seed
(PSNR / SSIM).

| config | peak VRAM | total time | KV ratio | KV encode | PSNR | SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| none (baseline) | 67.2 GiB | 59.7 s | 1.00× | — | — | — |
| entropack 8 bpp | 51.5 GiB | 68.9 s | 1.99× | 5.2 s | 23.8 dB | 0.913 |
| entropack 6 bpp | 47.5 GiB | 67.9 s | 2.65× | 5.3 s | 22.3 dB | 0.881 |
| entropack 4 bpp | 43.5 GiB | 67.3 s | 3.98× | 5.1 s | 21.2 dB | 0.846 |
| entropack 2 bpp | 39.5 GiB | 67.4 s | 7.97× | 4.9 s | 11.3 dB | 0.537 |
| svd rank=64 | 51.4 GiB | 85.0 s | 1.99× | 19.1 s | 9.0 dB | 0.282 |
| svd rank=32 | 43.5 GiB | 76.9 s | 3.98× | 11.9 s | 8.7 dB | 0.303 |

> `svd` calls `torch.svd_lowrank` directly; `cholesky_qr` runs the same randomized SVD with a
> Cholesky-QR range finder, which is 4–8× faster to encode on accelerators with slow batched QR
> (here: `[svd(rank=32)]` 69 s vs `[cholesky_qr(rank=32)]` 8.8 s; joint rank=1024: 96 s vs 27.5 s)
> with an identical compression ratio and image quality.
| svd rank=32 + ep 8 bpp | 39.5 GiB | 82.2 s | 7.92× | 15.5 s | 8.9 dB | 0.310 |
| svd rank=64 + ep 8 bpp | 43.5 GiB | 91.7 s | 3.96× | 23.3 s | 9.3 dB | 0.288 |

The encoder costs ~5 s once and the ~5000 temporary decodes cost ~2.3 s across the run; because the
entropack path keeps the decoded tensors contiguous, the decode-path concatenation stays as cheap as in
the uncompressed run. EntroPack 6 bpp is a good default for this workload: it removes ~20 GiB of VRAM while keeping the
composition, identities and outfits of the uncompressed result. `entropack 4 bpp` still produces a
coherent photo but starts changing identities.

Low-rank SVD is not competitive here. The per-head singular-value spectrum of RoPE-rotated keys and
of the value projections decays slowly, so truncating at a useful ratio removes information that
attention is sensitive to: `rank=64` keeps 94% of the energy yet only matches a 2× ratio, and
`rank=32` changes the output beyond recognition. EntroPack codes the same tensors at 6 bpp with 2%
relative error, versus 40% for `svd rank=32` at a comparable ratio. Low-rank compression remains
available for tensors whose spectrum does decay.
