# `diffsynth.core.attention`: 注意力机制实现

`diffsynth.core.attention` 提供了注意力机制实现的路由机制，根据 `Python` 环境中的可用包和[环境变量](/docs/zh/Pipeline_Usage/Environment_Variables.md#diffsynth_attention_implementation)自动选择高效的注意力机制实现。

## 注意力机制

注意力机制是在论文[《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)中提出的模型结构，在原论文中，注意力机制按照如下公式实现：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(
    \frac{QK^T}{\sqrt{d_k}}
\right)
V.
$$

在 `PyTorch` 中，可以用如下代码实现：
```python
import torch

def attention(query, key, value):
    scale_factor = 1 / query.size(-1)**0.5
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value

query = torch.rand(32, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
output_1 = attention(query, key, value)
```

其中 `query`、`key`、`value` 的维度是 $(b, n, s, d)$：
* $b$：Batch size
* $n$: Attention head 的数量
* $s$: 序列长度
* $d$: 每个 Attention head 的维数

这部分计算是不包含任何可训练参数的，现代 transformer 架构的模型会在进行这一计算前后经过 Linear 层，本文讨论的“注意力机制”不包含这些计算，仅包含以上代码的计算。

## 更高效的实现

注意到，注意力机制中 Attention Score（公式中的 $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$，代码中的 `attn_weight`）的维度为 $(b, n, s, s)$，其中序列长度 $s$ 通常非常大，这导致计算的时间和空间复杂度达到平方级。以图像生成模型为例，图像的宽度和高度每增加到 2 倍，序列长度增加到 4 倍，计算量和显存需求增加到 16 倍。为了避免高昂的计算成本，需采用更高效的注意力机制实现，包括
* Flash Attention 3：[GitHub](https://github.com/Dao-AILab/flash-attention)、[论文](https://arxiv.org/abs/2407.08608)
* Flash Attention 2：[GitHub](https://github.com/Dao-AILab/flash-attention)、[论文](https://arxiv.org/abs/2307.08691)
* Sage Attention：[GitHub](https://github.com/thu-ml/SageAttention)、[论文](https://arxiv.org/abs/2505.11594)
* xFormers：[GitHub](https://github.com/facebookresearch/xformers)、[文档](https://facebookresearch.github.io/xformers/components/ops.html#module-xformers.ops)
* PyTorch：[GitHub](https://github.com/pytorch/pytorch)、[文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

如需调用除 `PyTorch` 外的其他注意力实现，请按照其 GitHub 页面的指引安装对应的包。`DiffSynth-Studio` 会自动根据 Python 环境中的可用包路由到对应的实现上，也可通过[环境变量](/docs/zh/Pipeline_Usage/Environment_Variables.md#diffsynth_attention_implementation)控制。

```python
from diffsynth.core.attention import attention_forward
import torch

def attention(query, key, value):
    scale_factor = 1 / query.size(-1)**0.5
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value

query = torch.rand(32, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
output_1 = attention(query, key, value)
output_2 = attention_forward(query, key, value)
print((output_1 - output_2).abs().mean())
```

请注意，加速的同时会引入误差，但在大多数情况下误差是可以忽略不计的。

## 开发者导引

在为 `DiffSynth-Studio` 接入新模型时，开发者可自行决定是否调用 `diffsynth.core.attention` 中的 `attention_forward`，但我们期望模型能够尽可能优先调用这一模块，以便让新的注意力机制实现能够在这些模型上直接生效。

## 最佳实践

**在大多数情况下，我们建议直接使用 `PyTorch` 原生的实现，无需安装任何额外的包。** 虽然其他注意力机制实现可以加速，但加速效果是较为有限的，在少数情况下会出现兼容性和精度不足的问题。

此外，高效的注意力机制实现会逐步集成到 `PyTorch` 中，`PyTorch` 的 `2.9.0` 版本中的 `scaled_dot_product_attention` 已经集成了 Flash Attention 2。我们仍在 `DiffSynth-Studio` 提供这一接口，是为了让一些激进的加速方案能够快速走向应用，尽管它们在稳定性上还需要时间验证。
