# KV Cache 压缩

`QwenImage21Pipeline` 会在第一个去噪步把条件前缀（提示词与参考图 token）写成 KV Cache，并在后续每一步复用。这份缓存经常是整个 pipeline 中最大的常驻张量：使用 8 张 1024×1024 参考图时，每个 Transformer block 的前缀长度为 32510 个 token，单个 CFG 分支就是 16.3 GiB，正负两个分支合计 32.5 GiB，超过 30 GiB 的模型权重。

`kv_compress_config` 让这份缓存以压缩形式保存：缓存首次产生时压缩一次，之后每次被读取时临时解压。

## 快速上手

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
    kv_compress_config=[{"method": "entropack", "target_bpp": 6.0}],
)
image.save("image.png")
print(pipe.last_kv_compression_stats)
```

`kv_compress_config` 是**按顺序执行的压缩方案列表**（单个 dict 视为只有一级）；默认为 `None`，即不压缩。`template_kv_cache` 与 `negative_template_kv_cache` 是另一套机制，不受影响。

## 压缩方案

| 方案 | 配置示例 | 存储内容 |
| --- | --- | --- |
| `svd` | `{"method": "svd", "rank": 64}` | `torch.svd_lowrank` 得到的低秩因子 `U`、`S`、`V` |
| `cholesky_qr` | `{"method": "cholesky_qr", "rank": 1024}` | 同样的低秩因子，用 Cholesky-QR 正交化的范围搜索 |
| `entropack` | `{"method": "entropack", "target_bpp": 4.0}` | EntroPack 熵编码后的原始数值 |

方案通过列表叠加：`[{"method": "svd", "rank": 1024}, {"method": "entropack", "target_bpp": 8.0}]`
表示先 SVD 再对因子做熵编码，解压时倒序执行。新增方案只需继承 `Scheme` 并调用 `register_scheme`。

### SVD

缓存张量被视作 `batch × heads` 个 `[前缀长度, head_dim]` 矩阵，用快速随机化 SVD 分解。`rank` 指定保留的奇异三元组数量，`rank_ratio` 按 `min(m, n)` 的比例指定，`energy` 则保留能量占比达到要求的最小秩。

`engine` 选择实现：`"svd_lowrank"` 调用 `torch.svd_lowrank`（PyTorch 自带的随机化近似 SVD），`"randomized"` 用批量矩阵乘与 Cholesky-QR 实现同样的随机化算法（在批量 QR 较慢的加速卡上快很多），`"auto"` 在 CUDA 上选择 `"randomized"`。

### Cholesky-QR（`cholesky_qr`）

与 `svd` 相同的随机化 SVD，但范围搜索用两次 Cholesky-QR 正交化代替 `torch.linalg.qr`。
结果与压缩率一致，在批量 QR 较慢的加速卡上编码快 4–8 倍（head rank=32：0.07 s vs 0.55 s/张量；
joint rank=1024：0.23 s vs 0.85 s）。参数与 `svd` 完全相同。

### EntroPack

[EntroPack](https://github.com/modelscope/entropack) 提供 GPU 熵编码。`scheme="lossy"` 按 `target_bpp`（0.001–11 bit/元素）做率控制有损压缩，`scheme="tile_ans"` 与 `scheme="dfloat11"` 是无损压缩。需要安装 `entropack` 以及可用的 CuPy：

```bash
pip install "entropack[cuda13]"   # cupy-cuda13x
# 若显卡厂商提供了自己的 CuPy 构建，请安装对应版本
```

EntroPack 的有损压缩按行分配量化尺度，行非常窄时精度会明显下降，因此编解码器会把连续多行合并到 `row_width`（默认 128）个元素；元素数少于 `min_numel`（默认 4096）的张量直接原样存储。

### SVD + EntroPack

叠加模式对 SVD 因子再做 EntroPack，压缩率是两级收益的乘积。但它同时继承了 SVD 的截断误差：同样的显存预算下，单独使用 EntroPack 通常保留更多信息（见下方实测）。


## 逐 head 与联合 SVD（`layout`）

KV 张量形状是 `[batch, seq, heads, head_dim]`，`layout` 决定 SVD 作用在哪个矩阵上：

* `"head"`（默认）：每个 `(batch, head)` 一个 `[seq, head_dim]` 矩阵，SVD 作用在序列轴与 head 维度轴之间。
* `"joint"`：每个 batch 一个 `[seq, heads * head_dim]` 矩阵，把各 head 的通道拼到列轴上，
  SVD 作用在序列轴与展平后的 head/dim 轴之间。

在这份缓存上联合布局明显更有效：联合矩阵的奇异值谱衰减快得多。保留约 25% 的维度时
（4096 中的 rank 1024）能量保留 98.7%，而逐 head 保留 25%（128 中的 rank 32）只有 84%。
真实前缀张量实测：

| 配置 | KV 相对误差 | 压缩率 |
| --- | --- | --- |
| 逐 head SVD rank=32 | 0.399 | 3.98× |
| 联合 SVD rank=1024 | **0.112** | 3.55× |
| 逐 head SVD rank=64 | 0.237 | 1.99× |
| 联合 SVD rank=2048 | **0.049** | 1.78× |

生图效果（8 张图、1024²、40 步）：

| 配置 | 显存峰值 | 总耗时 | 压缩率 | PSNR | SSIM | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| 联合 rank=2048 | 53.4 GiB | 151.5 s | 1.78× | 16.3 dB | 0.727 | 与基线几乎一致 |
| 联合 rank=1024 | 44.5 GiB | 92.3 s | 3.55× | 13.2 dB | 0.594 | 画面合理，人物漂移 |
| 联合 rank=512 | 40.0 GiB | 73.3 s | 7.11× | 11.3 dB | 0.517 | 画面合理但有蜡感伪影 |
| 逐 head rank=64 | 51.4 GiB | 85.0 s | 1.99× | 9.0 dB | 0.282 | 崩坏 |
| 逐 head rank=32 | 43.5 GiB | 76.9 s | 3.98× | 8.7 dB | 0.303 | 崩坏 |

> `svd` 直接调用 `torch.svd_lowrank`；`cholesky_qr` 是同一随机化 SVD，但范围搜索用 Cholesky-QR 正交化，
> 在批量 QR 较慢的加速卡上编码快 4–8 倍（本机：`[svd(rank=32)]` 69 s vs `[cholesky_qr(rank=32)]` 8.8 s；
> joint rank=1024：96 s vs 27.5 s），压缩率与画质完全一致。

联合布局让低秩 KV 压缩变得可用，但在同样的显存预算下仍不如 EntroPack
（EntroPack 4 bpp：3.98×、误差 0.081、编码仅 5.6 s，而联合 rank=1024 编码要 27.5 s）。

## 统计信息

`pipe.last_kv_compression_stats` 给出最近一次调用的缓存统计：

```json
{"method": "entropack(lossy@6.0bpp)", "compress_calls": 128, "decompress_calls": 4992,
 "original_gib": 31.746, "stored_gib": 11.984, "compression_ratio": 2.649,
 "effective_bpp": 6.04, "compress_seconds": 5.45, "decompress_seconds": 2.67}
```

`compress_seconds` 是第一步的一次性编码开销，`decompress_seconds` 是后续所有步临时解压的累计耗时。

## 通用张量接口

同一套编解码器可以压缩任意张量，而不只是 KV Cache：

```python
from diffsynth.core.context_compress import compress_tensor, decompress_tensor

store = compress_tensor(tensor, [
    {"method": "svd", "rank": 32},
    {"method": "entropack", "target_bpp": 8.0},
])
print(store.nbytes, store.compression_ratio, store.bits_per_element)
dense = decompress_tensor(store)
store = store.to("cpu")   # 只移动压缩后的存储，不解码
```

代码位于 `diffsynth/core/context_compress`。

## 实测

8 张参考图、1024×1024、40 步、`cfg_scale=4`，单张 PPU（96 GiB）。画质对比的是同一 seed 未压缩的结果（PSNR / SSIM）。

| 配置 | 显存峰值 | 总耗时 | KV 压缩率 | KV 编码耗时 | PSNR | SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| 不压缩（基线） | 67.2 GiB | 59.7 s | 1.00× | — | — | — |
| entropack 8 bpp | 51.5 GiB | 68.9 s | 1.99× | 5.2 s | 23.8 dB | 0.913 |
| entropack 6 bpp | 47.5 GiB | 67.9 s | 2.65× | 5.3 s | 22.3 dB | 0.881 |
| entropack 4 bpp | 43.5 GiB | 67.3 s | 3.98× | 5.1 s | 21.2 dB | 0.846 |
| entropack 2 bpp | 39.5 GiB | 67.4 s | 7.97× | 4.9 s | 11.3 dB | 0.537 |
| svd rank=64 | 51.4 GiB | 85.0 s | 1.99× | 19.1 s | 9.0 dB | 0.282 |
| svd rank=32 | 43.5 GiB | 76.9 s | 3.98× | 11.9 s | 8.7 dB | 0.303 |
| svd rank=32 + ep 8 bpp | 39.5 GiB | 82.2 s | 7.92× | 15.5 s | 8.9 dB | 0.310 |
| svd rank=64 + ep 8 bpp | 43.5 GiB | 91.7 s | 3.96× | 23.3 s | 9.3 dB | 0.288 |

编码是一次性开销（约 5 s），整轮约 5000 次临时解压合计约 2.3 s；entropack 路径的解码结果保持连续，解码端的 `cat` 与不压缩时同样快。在这个场景下推荐 `entropack 6 bpp`：省下约 20 GiB 显存，同时基本保持未压缩结果的构图、人物与服装。`entropack 4 bpp` 仍能生成合理的照片，但人物开始发生变化。

低秩 SVD 在这里并不占优。经过 RoPE 的 key 与 value 投影，其按 head 的奇异值谱衰减很慢，在有效压缩率下截断会丢掉 attention 敏感的信息：`rank=64` 保留了 94% 能量却只相当于 2× 压缩率，`rank=32` 则已经无法保持画面。相同压缩率下，EntroPack 6 bpp 的相对误差是 2%，而 `svd rank=32` 是 40%。对于奇异值谱确实快速衰减的张量，低秩压缩依然可用。
