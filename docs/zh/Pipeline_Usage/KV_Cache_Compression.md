# KV-Cache 压缩

部分模型采用 KV-Cache 存储输入的文本和图像信息，在多图编辑等任务中占用极大的显存。DiffSynth-Studio 框架提供通用的 KV-Cache 压缩接口，能够大幅减少 KV-Cache 所占用的显存。

> 目前 KV-Cache 压缩仅适配 Qwen-Image-2.1 和 Qwen-Image-2.1-Turbo。

## 启用 KV-Cache 压缩

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
    "生成这些角色的合影",
    edit_image=[Image.open(path) for path in image_paths],
    seed=1,
    cfg_scale=4,
    kv_compress_config=[{"method": "entropack", "target_bpp": 4.0}],
)
image.save("image.png")
```

`kv_compress_config` 是**按顺序执行的压缩方案列表**（单个 dict 视为只有一级）；默认为 `None`，即不压缩。

## 压缩方案

| 方案 | 配置示例 | 存储内容 | 使用说明 |
| --- | --- | --- | --- |
| `entropack` | `{"method": "entropack", "target_bpp": 4.0}` | EntroPack 熵编码后的原始数值 | 速度最快，效果最好，推荐使用 |
| `svd` | `{"method": "svd", "rank": 64, "layout": "joint"}` | `torch.svd_lowrank` 得到的低秩因子 `U`、`S`、`V` | 速度慢，不建议使用，目前仅用于研究 |
| `cholesky_qr` | `{"method": "cholesky_qr", "rank": 1024, "layout": "joint"}` | 同样的低秩因子，用 Cholesky-QR 正交化的范围搜索 | 速度中等，不建议使用，目前仅用于研究 |

支持 SVD + EntroPack，或者 Cholesky-QR + EntroPack 融合方案，先做矩阵分解，再做量化压缩，压缩率是两级收益的乘积，但存在误差累积。融合方案可通过列表形式填写 config，例如 `[{"method": "cholesky_qr", "rank": 1024, "layout": "joint"}, {"method": "entropack", "target_bpp": 8.0}]`。

## 效果一览

测试条件：Qwen-Image-2.1、8 张参考图、1024×1024、40 步、`cfg_scale=4`，单张 96 GiB PPU 加速卡。PSNR / SSIM 均与“不压缩”的结果对比。

| 方案 | 计算时间 | 峰值显存 | PSNR | SSIM |
| --- | --- | --- | --- | --- |
| 不压缩 | 60.1 s | 67.2 GiB | — | — |
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

现阶段，`entropack` 是效果最好的方案。

## 通用张量压缩接口

同一套编解码器可以压缩任意张量，而不只是 KV Cache。如果你希望将这个模块用于其他项目和目的，请参考以下代码：

```python
from diffsynth.core.context_compress import compress_tensor, decompress_tensor

store = compress_tensor(tensor, [
    {"method": "svd", "rank": 32},
    {"method": "entropack", "target_bpp": 8.0},
])
print(store.nbytes, store.compression_ratio, store.bits_per_element)
dense = decompress_tensor(store)
```
