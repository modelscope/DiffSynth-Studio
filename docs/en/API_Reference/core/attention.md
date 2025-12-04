# `diffsynth.core.attention`: Attention Mechanism Implementation

`diffsynth.core.attention` provides routing mechanisms for attention mechanism implementations, automatically selecting efficient attention implementations based on available packages in the `Python` environment and [environment variables](/docs/en/Pipeline_Usage/Environment_Variables.md#diffsynth_attention_implementation).

## Attention Mechanism

The attention mechanism is a model structure proposed in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). In the original paper, the attention mechanism is implemented according to the following formula:

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(
    \frac{QK^T}{\sqrt{d_k}}
\right)
V.
$$

In `PyTorch`, it can be implemented with the following code:
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

The dimensions of `query`, `key`, and `value` are $(b, n, s, d)$:
* $b$: Batch size
* $n$: Number of attention heads
* $s$: Sequence length
* $d$: Dimension of each attention head

This computation does not include any trainable parameters. Modern transformer architectures will pass through Linear layers before and after this computation, but the "attention mechanism" discussed in this article refers only to the computation in the above code, not including these calculations.

## More Efficient Implementations

Note that the dimension of the Attention Score in the attention mechanism ( $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ in the formula, `attn_weight` in the code) is $(b, n, s, s)$, where the sequence length $s$ is typically very large, causing the time and space complexity of computation to reach quadratic level. Taking image generation models as an example, when the width and height of the image increase to 2 times, the sequence length increases to 4 times, and the computational load and memory requirements increase to 16 times. To avoid high computational costs, more efficient attention mechanism implementations are needed, including:
* Flash Attention 3: [GitHub](https://github.com/Dao-AILab/flash-attention), [Paper](https://arxiv.org/abs/2407.08608)
* Flash Attention 2: [GitHub](https://github.com/Dao-AILab/flash-attention), [Paper](https://arxiv.org/abs/2307.08691)
* Sage Attention: [GitHub](https://github.com/thu-ml/SageAttention), [Paper](https://arxiv.org/abs/2505.11594)
* xFormers: [GitHub](https://github.com/facebookresearch/xformers), [Documentation](https://facebookresearch.github.io/xformers/components/ops.html#module-xformers.ops)
* PyTorch: [GitHub](https://github.com/pytorch/pytorch), [Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

To call attention implementations other than `PyTorch`, please follow the instructions on their GitHub pages to install the corresponding packages. `DiffSynth-Studio` will automatically route to the corresponding implementation based on available packages in the Python environment, or can be controlled through [environment variables](/docs/en/Pipeline_Usage/Environment_Variables.md#diffsynth_attention_implementation).

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

Please note that acceleration will introduce errors, but in most cases, the error is negligible.

## Developer Guide

When integrating new models into `DiffSynth-Studio`, developers can decide whether to call `attention_forward` in `diffsynth.core.attention`, but we expect models to prioritize calling this module as much as possible, so that new attention mechanism implementations can take effect directly on these models.

## Best Practices

**In most cases, we recommend directly using the native `PyTorch` implementation without installing any additional packages.** Although other attention mechanism implementations can accelerate, the acceleration effect is relatively limited, and in a few cases, compatibility and precision issues may arise.

In addition, efficient attention mechanism implementations will gradually be integrated into `PyTorch`. The `scaled_dot_product_attention` in `PyTorch` version 2.9.0 has already integrated Flash Attention 2. We still provide this interface in `DiffSynth-Studio` to allow some aggressive acceleration schemes to quickly move toward application, even though they still need time to be verified for stability.