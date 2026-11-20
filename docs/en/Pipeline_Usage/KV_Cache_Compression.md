# KV-Cache Compression

Some models use a KV-Cache to store the text and image information of the input, which takes up a
large amount of VRAM in multi-image editing and similar tasks. DiffSynth-Studio provides a general
KV-Cache compression interface that greatly reduces the VRAM used by the KV-Cache.

> KV-Cache compression currently only supports Qwen-Image-2.1 and Qwen-Image-2.1-Turbo.

## Enabling KV-Cache Compression

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
    kv_compress_config=[{"method": "entropack", "target_bpp": 4.0}],
)
image.save("image.png")
```

`kv_compress_config` is a **list of compression stages executed in order** (a single dict is treated
as a one-stage list); the default `None` means no compression.

## Compression Stages

| Stage | Example config | Stored content | Usage |
| --- | --- | --- | --- |
| `entropack` | `{"method": "entropack", "target_bpp": 4.0}` | Original values after EntroPack entropy coding | Fastest and best quality, recommended |
| `svd` | `{"method": "svd", "rank": 64, "layout": "joint"}` | Low-rank factors `U`, `S`, `V` from `torch.svd_lowrank` | Slow, not recommended, for research only |
| `cholesky_qr` | `{"method": "cholesky_qr", "rank": 1024, "layout": "joint"}` | The same low-rank factors, with a Cholesky-QR orthonormalized range finder | Medium speed, not recommended, for research only |

SVD + EntroPack and Cholesky-QR + EntroPack combinations are supported: the matrix is decomposed
first and then quantized, so the compression ratio is the product of both stages, at the cost of
accumulated error. A combined config is written as a list, for example
`[{"method": "cholesky_qr", "rank": 1024, "layout": "joint"}, {"method": "entropack", "target_bpp": 8.0}]`.

## Results

Test conditions: Qwen-Image-2.1, 8 reference images, 1024×1024, 40 steps, `cfg_scale=4`, a single
96 GiB PPU accelerator. PSNR / SSIM are measured against the uncompressed result.

| Config | Time | Peak VRAM | PSNR | SSIM |
| --- | --- | --- | --- | --- |
| No compression | 60.1 s | 67.2 GiB | — | — |
| `entropack` 8 bpp | 69.0 s | 51.5 GiB | 23.76 dB | 0.913 |
| `entropack` 6 bpp | 68.0 s | 47.5 GiB | 22.29 dB | 0.881 |
| `entropack` 4 bpp | 67.1 s | 43.5 GiB | 21.22 dB | 0.846 |
| `entropack` 2 bpp | 67.2 s | 39.5 GiB | 11.31 dB | 0.537 |
| `svd` rank=1024 | 160.7 s | 44.4 GiB | 13.19 dB | 0.595 |
| `svd` rank=2048 | 275.0 s | 53.4 GiB | 16.20 dB | 0.725 |
| `cholesky_qr` rank=1024 | 92.0 s | 44.5 GiB | 13.18 dB | 0.594 |
| `cholesky_qr` rank=2048 | 151.2 s | 53.4 GiB | 16.26 dB | 0.727 |
| `svd` 1024 + `entropack` 6 bpp | 164.0 s | 38.9 GiB | 12.60 dB | 0.576 |
| `cholesky_qr` 1024 + `entropack` 6 bpp | 95.2 s | 38.9 GiB | 12.21 dB | 0.540 |

At present, `entropack` gives the best results.

## General Tensor Compression API

The same codecs can compress arbitrary tensors, not only the KV-Cache. If you want to use this
module in another project or for another purpose, see the code below:

```python
from diffsynth.core.context_compress import compress_tensor, decompress_tensor

store = compress_tensor(tensor, [
    {"method": "svd", "rank": 32},
    {"method": "entropack", "target_bpp": 8.0},
])
print(store.nbytes, store.compression_ratio, store.bits_per_element)
dense = decompress_tensor(store)
```
