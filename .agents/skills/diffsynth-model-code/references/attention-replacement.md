# Attention 替换指南

**适用对象**：DiT 及其他噪声预测模型中的 self-attention / cross-attention 计算。

## 核心原则

DiffSynth 提供统一的 `attention_forward` 函数，自动选择最优 attention 后端（flash_attention_4 → flash_attention_3 → flash_attention_2 → sage_attention → xformers → torch_sdpa）。接入新模型时，应将目标库的 attention 实现替换为此统一接口。

## `attention_forward` 权威实现

文件位置：`diffsynth/core/attention/attention.py`

```python
def attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    attn_mask=None,
    scale=None,
    is_causal=False,
    compatibility_mode=False,
    window_size=None,
    use_flex=False,
    score_mod=None,
):
```

**⚠️ `compatibility_mode` 排在 `is_causal` 之后**，做位置传参时容易错位，建议一律用关键字传参。

**自动后端选择**（通过 `DIFFSYNTH_ATTENTION_IMPLEMENTATION` 环境变量或自动检测）：

| 优先级 | 后端 | 依赖 | 支持 pattern | 支持 custom mask |
|--------|------|------|-------------|-----------------|
| 1 | flash_attention_4 | `flash_attn.cute` | `b n s d` → 内部转 `b s n d` | 否 |
| 2 | flash_attention_3 | `flash_attn_interface` | `b n s d` → 内部转 `b s n d` | 否 |
| 3 | flash_attention_2 | `flash_attn` | `b n s d` → 内部转 `b s n d` | 否 |
| 4 | sage_attention | `sageattention` | `b n s d`（无内部转置） | 否 |
| 5 | xformers | `xformers` | `b n s d` → 内部转 `b s n d` | 否 |
| 6 | torch_sdpa | torch（无依赖） | `b n s d` | 是 |

**关键实现细节**：
- `rearrange_qkv` 内部使用 einops 将 QKV 从 pattern 转成内部所需格式
- flash_attention_2/3/4 内部需要 `b s n d`，会自动从 `b n s d` 转换
- sage_attention 内部直接接受 `b n s d`
- torch_sdpa 内部直接接受 `b n s d`
- `compatibility_mode=True` 或有 `attn_mask` 时走 torch 分支，该分支内再细分：`use_flex` 或给了 `score_mod` → `flex_attention`；给了 `window_size` → `torch_sdpa_sliding_window`（此时 `is_causal` 必须为 False 且 `attn_mask` 必须为 None）；否则 → `torch_sdpa`
- sage_attention / xformers 遇到 `window_size` 或 `is_causal` 时会回退到 `compatibility_mode=True`

## Pattern 说明

- `"b n s d"` — batch, heads, seq, dim（内部 torch_sdpa/sage_attention 原生格式）
- `"b s n d"` — batch, seq, heads, dim（**DiffSynth 现有模型全部采用此格式**）
- `"b s (n d)"` — batch, seq, (heads * dim)（anima_dit 特殊格式，扁平化 heads*dim）

## 替换步骤

### 步骤 1：替换 import

```python
# 删除目标库原有的 attention import 或实现
# from .attention import SomeCustomAttention  ← 删除

# 替换为
from ..core.attention import attention_forward
```

### 步骤 2：定位 attention 计算点

在目标库的 attention 类/方法中，找到实际的 Q @ K @ V 计算。常见形式：
- `F.scaled_dot_product_attention(q, k, v)`
- `q @ k.transpose(-2, -1) * scale → softmax → @ v`
- 自定义的 `xformers.ops.memory_efficient_attention(q, k, v)`

### 步骤 3：确保 QKV 格式

`attention_forward` 内部通过 einops `rearrange` 自动处理 pattern 转换。QKV 的最后一个维度可以是：
- **分离的 heads 和 dim**：`(..., heads, dim)` → 使用标准 pattern 如 `"b s n d"`
- **扁平化的 heads*dim**：`(..., heads * dim)` → 使用 `"b s (n d)"`，需要在 `dims` 中传入 `n` 和 `d` 的值

**DiffSynth 现有模型统一采用 `"b s n d"` 格式**（batch, seq, heads, dim），通过 `unflatten(-1, (heads, dim))` 从线性层输出转换得到。这是推荐的做法：

```python
# 线性层输出 shape: (batch, seq, heads * dim)
# 使用 unflatten 拆分为 (batch, seq, heads, dim)
q = q.unflatten(-1, (self.heads, self.dim_head))
k = k.unflatten(-1, (self.heads, self.dim_head))
v = v.unflatten(-1, (self.heads, self.dim_head))
```

### 步骤 4：替换为 attention_forward 调用

```python
# 替换前（目标库原始实现）
attn_output = F.scaled_dot_product_attention(
    query_states, key_states, value_states,
    attn_mask=attention_mask,
    scale=self.scaling,
)

# 替换后（推荐格式 "b s n d"）
attn_output = attention_forward(
    query_states, key_states, value_states,
    q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
    attn_mask=attention_mask,
)
```

**注意**：scale 参数通常不需要显式传递，`attention_forward` 会使用 torch SDPA 的默认行为（`1/sqrt(dim)`）。只有在目标库使用了非标准 scale 时才需要传入。

### 步骤 5：输出 reshape

`attention_forward` 返回的 output 格式与 `out_pattern` 一致。如果后续代码需要扁平化的 `(batch, seq, heads * dim)`，需要手动 `flatten`：

```python
# attention_forward 返回 (batch, seq, heads, dim)
hidden_states = attention_forward(...)
# flatten 为 (batch, seq, heads * dim) 以便后续 Linear 层处理
hidden_states = hidden_states.flatten(2, 3)
```

### 步骤 6：删除原有 attention 实现

删除目标库文件中与 `attention_forward` 功能重复的类或函数。

## 现有模型参考

以下所有模型均采用 `"b s n d"` pattern，通过 `unflatten` 从线性层输出转换得到 QKV。

| 模型文件 | Pattern | QKV 准备 | 特殊处理 |
|---------|---------|---------|---------|
| `z_image_dit.py` | `b s n d` | `unflatten(-1, (num_heads, -1))` + RoPE + norm | `attn_mask` 走 torch_sdpa |
| `flux2_dit.py` | `b s n d` | `unflatten(-1, (heads, -1))` + RoPE + norm | encoder_hidden_states 拼接 + token split |
| `ltx2_dit.py` | `b s n d` | `unflatten(-1, (heads, dim_head))` + RoPE + norm | mask 走 torch_sdpa |
| `anima_dit.py` | `b s (n d)` | `rearrange` 展平 | 自定义 wrapper 函数 |

### z_image_dit.py 标准模式（最简参考）

```python
from ..core.attention import attention_forward

class SelfAttention(nn.Module):
    def forward(self, hidden_states, freqs_cis, attention_mask):
        # 1. 线性投影
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # 2. unflatten 拆分为 (batch, seq, heads, dim)
        query = query.unflatten(-1, (self.num_heads, -1))
        key = key.unflatten(-1, (self.num_heads, -1))
        value = value.unflatten(-1, (self.num_heads, -1))

        # 3. 可选的 norm 和 RoPE
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)
        if freqs_cis is not None:
            query = self.apply_rotary_emb(query, freqs_cis)
            key = self.apply_rotary_emb(key, freqs_cis)

        # 4. 调用 attention_forward
        hidden_states = attention_forward(
            query, key, value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            attn_mask=attention_mask,
        )

        # 5. flatten 回 (batch, seq, heads * dim)
        hidden_states = hidden_states.flatten(2, 3)

        # 6. 输出投影
        return self.to_out(hidden_states)
```

### flux2_dit.py cross-attention 模式

```python
# encoder-hidden_states 与 hidden_states 的 KV 拼接
if attn.added_kv_proj_dim is not None:
    encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
    encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
    encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

    encoder_query = attn.norm_added_q(encoder_query)
    encoder_key = attn.norm_added_k(encoder_key)

    query = torch.cat([encoder_query, query], dim=1)
    key = torch.cat([encoder_key, key], dim=1)
    value = torch.cat([encoder_value, value], dim=1)

# 统一 attention
hidden_states = attention_forward(
    query, key, value,
    q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
)

# 拆分回 encoder 和 decoder 部分
encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
    [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
)
```

## 特殊场景处理

### GQA / MQA

`attention_forward` 现已原生支持 Grouped Query / Multi Query Attention，**无需在调用前手动展开 key/value 的 head**。直接把 head 数少于 Q 的 K/V 传入即可。

```python
# K/V 的 head 数可以少于 Q，直接传入，无需手动 expand/repeat
attn_output = attention_forward(
    query_states, key_states, value_states,
    q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
)
```

**内部机制**（`torch_sdpa`，见 `diffsynth/core/attention/attention.py`）：QKV 统一转成 `b n s d` 后，检测到 Q 与 K/V 的 head 数（`n`）不一致即自动处理 GQA——
- 较新版本 torch（`scaled_dot_product_attention` 支持 `enable_gqa`）：直接传 `enable_gqa=True`
- 低版本 torch：自动 `repeat` 展开 K/V 的 head 后再计算

> GQA 自动展开在 `torch_sdpa` 后端实现（`compatibility_mode=True`、传入 `attn_mask` 或无其他后端时走此路径）。flash_attention 后端原生支持不同 head 数；若使用 sage / xformers 后端遇到 head 数不匹配，可设 `compatibility_mode=True` 强制走 torch_sdpa。

### Custom Attention Mask

如果目标库使用了非标准的 attention mask，设置 `compatibility_mode=True` 强制走 torch SDPA（flash attention 不支持 arbitrary mask）。

```python
attn_output = attention_forward(
    q, k, v,
    q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
    attn_mask=custom_mask,
    compatibility_mode=True,  # 有 custom mask 时必须设置
)
```

**内部机制**：当 `attn_mask is not None` 或 `compatibility_mode=True` 时，直接路由到 `torch_sdpa`，绕过 flash/sage/xformers 后端。

### RoPE（旋转位置编码）

RoPE 在 `attention_forward` **之前**应用到 Q 和 K 上。参考各现有模型的实现：

```python
# z_image_dit / flux2_dit 风格
if freqs_cis is not None:
    query = self.apply_rotary_emb(query, freqs_cis)
    key = self.apply_rotary_emb(key, freqs_cis)

# ltx2_dit 风格
if pe is not None:
    q = apply_rotary_emb(q, pe, self.rope_type)
    k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
```

### Scale 参数

对应目标库的 `softmax_scale` 或 `1 / sqrt(dim_head)`。大多数情况下不需要显式传递 scale，`attention_forward` 会使用 torch SDPA 的默认行为。仅在目标库使用了非标准 scale 时才传入。

### AnimaDiT 扁平化模式

当 QKV 以扁平化的 `(batch, seq, heads * dim)` 格式传入时：

```python
from ..core.attention import attention_forward
from einops import rearrange

def _attention(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D):
    # 转换为 (batch, heads, seq, dim) 格式
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(
        q_B_S_H_D.shape[0], q_B_S_H_D.shape[-2], -1, q_B_S_H_D.shape[-1])
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(
        k_B_S_H_D.shape[0], k_B_S_H_D.shape[-2], -1, k_B_S_H_D.shape[-1])
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(
        v_B_S_H_D.shape[0], v_B_S_H_D.shape[-2], -1, v_B_S_H_D.shape[-1])
    # 输出保持扁平化
    return attention_forward(q_B_H_S_D, k_B_H_S_D, v_B_H_S_D, out_pattern="b s (n d)")
```

## 相关参考

- Gradient Checkpointing 替换 → [gradient-checkpointing.md](./gradient-checkpointing.md)
- 自有源码接入 → [model-file-own-source.md](./model-file-own-source.md)
