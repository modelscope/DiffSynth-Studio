# Parameter Count Audit: ControlNet LoRA vs Alternatives

**Date**: 2026-03-31
**Verified**: All counts validated against actual PyTorch model instantiation.

---

## Summary

| Configuration | Trainable Params | vs CN LoRA |
|--------------|----------------:|----------:|
| **ControlNet LoRA (rank 32)** | **40,304,640 (40.3M)** | **1.0x** |
| Hypothetical DiT LoRA (rank 32) | 153,157,632 (153.2M) | 3.80x |
| Full ControlNet fine-tuning | 3,301,925,888 (3.30B) | 81.9x |
| Full FLUX.1-dev DiT | ~12,000,000,000 (~12B) | ~298x |

**Key takeaways**:
- CN LoRA is **1.22%** of the full ControlNet
- CN LoRA is **3.8x smaller** than hypothetical DiT LoRA (exact block ratio: 15 vs 57)
- This is a core design decision: adapt ControlNet (condition pathway) with minimal parameters while keeping the DiT's generative prior completely frozen

---

## 1. Architecture Constants

| Symbol | Value | Source |
|--------|-------|--------|
| dim | 3072 | `FluxControlNet.__init__` / `FluxDiT.__init__` |
| num_heads | 24 | Same |
| head_dim | 128 | dim / num_heads |
| LoRA rank (r) | 32 | `train_scene_aware_raw.sh --lora_rank 32` |
| CN joint blocks | 5 | `FluxControlNet(num_joint_blocks=5)` |
| CN single blocks | 10 | `FluxControlNet(num_single_blocks=10)` |
| DiT joint blocks | 19 | `FluxDiT` |
| DiT single blocks | 38 | `FluxDiT` |

---

## 2. LoRA Parameter Formula

For a LoRA adapter of rank `r` on `Linear(in_dim, out_dim)`:

```
LoRA params = r × (in_dim + out_dim)
```

This adds two low-rank matrices: A ∈ R^{in_dim × r} and B ∈ R^{r × out_dim}.

---

## 3. LoRA Target Modules

From `train_scene_aware_raw.sh --lora_target_modules`:

```
a_to_qkv, b_to_qkv, ff_a.0, ff_a.2, ff_b.0, ff_b.2, a_to_out, b_to_out,
proj_out, norm.linear, norm1_a.linear, norm1_b.linear, to_qkv_mlp
```

13 target name patterns. These match by suffix against module names in the model.

---

## 4. Joint Block LoRA (10 targets per block)

Architecture: `FluxJointTransformerBlock(dim=3072, num_attention_heads=24)`

Contains: `AdaLayerNorm` (norm1_a, norm1_b), `FluxJointAttention` (attn), two feed-forward networks (ff_a, ff_b).

### AdaLayerNorm linear dimension

```python
# general_modules.py line 121
self.linear = torch.nn.Linear(dim, dim * [[6, 2][single], 9][dual])
```

Default (single=False, dual=False): `Linear(dim, dim × 6)` = `Linear(3072, 18432)`

The 6 outputs are: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.

### Per-target breakdown

| # | Target | Layer class | Linear(in, out) | LoRA: r × (in + out) |
|---|--------|------------|-----------------|--------------------:|
| 1 | `attn.a_to_qkv` | FluxJointAttention | (3072, 9216) | 393,216 |
| 2 | `attn.b_to_qkv` | FluxJointAttention | (3072, 9216) | 393,216 |
| 3 | `attn.a_to_out` | FluxJointAttention | (3072, 3072) | 196,608 |
| 4 | `attn.b_to_out` | FluxJointAttention | (3072, 3072) | 196,608 |
| 5 | `ff_a.0` | Feed-forward A up-proj | (3072, 12288) | 491,520 |
| 6 | `ff_a.2` | Feed-forward A down-proj | (12288, 3072) | 491,520 |
| 7 | `ff_b.0` | Feed-forward B up-proj | (3072, 12288) | 491,520 |
| 8 | `ff_b.2` | Feed-forward B down-proj | (12288, 3072) | 491,520 |
| 9 | `norm1_a.linear` | AdaLayerNorm | (3072, 18432) | 688,128 |
| 10 | `norm1_b.linear` | AdaLayerNorm | (3072, 18432) | 688,128 |
| | | | **Per joint block** | **4,521,984** |

### Derivation of linear dimensions

- `a_to_qkv` / `b_to_qkv`: projects to Q, K, V concatenated → dim × 3 = 9216
- `a_to_out` / `b_to_out`: attention output projection → dim = 3072
- `ff_*.0`: GELU MLP up-projection → dim × 4 = 12288
- `ff_*.2`: GELU MLP down-projection → 12288 back to 3072
- `norm1_*.linear`: AdaLayerNorm with 6 modulation outputs → dim × 6 = 18432

---

## 5. Single Block LoRA (3 targets per block)

Architecture: `FluxSingleTransformerBlock(dim=3072, num_attention_heads=24)`

Uses fused projections: QKV and MLP input are combined into one linear layer, and attention output and MLP output are combined into one output projection.

```python
# flux_dit.py line 213
self.to_qkv_mlp = torch.nn.Linear(dim, dim * (3 + 4))   # QKV(3) + MLP_in(4) = 7
self.proj_out = torch.nn.Linear(dim * 5, dim)            # attn_out(1) + MLP_out(4) = 5
```

### AdaLayerNormSingle linear dimension

```python
# flux_dit.py line 193
self.linear = torch.nn.Linear(dim, 3 * dim)  # Linear(3072, 9216)
```

3 outputs: shift_msa, scale_msa, gate_msa.

### Per-target breakdown

| # | Target | Layer class | Linear(in, out) | LoRA: r × (in + out) |
|---|--------|------------|-----------------|--------------------:|
| 1 | `to_qkv_mlp` | Fused QKV+MLP input | (3072, 21504) | 786,432 |
| 2 | `proj_out` | Fused attn+MLP output | (15360, 3072) | 589,824 |
| 3 | `norm.linear` | AdaLayerNormSingle | (3072, 9216) | 393,216 |
| | | | **Per single block** | **1,769,472** |

### Derivation of fused dimensions

- `to_qkv_mlp` out_dim: Q(3072) + K(3072) + V(3072) + MLP_in(4×3072) = 7 × 3072 = 21,504
- `proj_out` in_dim: attn_out(3072) + MLP_out(4×3072) = 5 × 3072 = 15,360

---

## 6. ControlNet LoRA Total

```
CN joint:   5 blocks × 4,521,984 params/block = 22,609,920
CN single: 10 blocks × 1,769,472 params/block = 17,694,720
                                          ──────────────────
CN LoRA total                              = 40,304,640
```

**80 LoRA adapters** total (5 × 10 joint targets + 10 × 3 single targets).

### Verified against model

```python
cn = FluxControlNet()
matched_params = 0
for name, module in cn.named_modules():
    if isinstance(module, torch.nn.Linear):
        for t in target_modules:
            if name.endswith(t):
                lora = 32 * (module.weight.shape[1] + module.weight.shape[0])
                matched_params += lora
# matched_params == 40,304,640 ✓
```

---

## 7. Hypothetical DiT LoRA Total

Same LoRA rank and target module names, applied to the DiT backbone instead:

```
DiT joint:  19 blocks × 4,521,984 params/block =  85,917,696
DiT single: 38 blocks × 1,769,472 params/block =  67,239,936
                                           ──────────────────
DiT LoRA total                              = 153,157,632
```

**Ratio: 153.2M / 40.3M = 3.80x** — exactly the block count ratio (57 / 15 = 3.80).

---

## 8. Full ControlNet Parameters

### Per joint block (full weights, not LoRA)

Includes weight matrices + bias terms: `Linear(in, out)` has `in × out + out` parameters.

| Module | Linear dims | Params |
|--------|------------|------:|
| `norm1_a.linear` | (3072, 18432) | 56,641,536 |
| `norm1_b.linear` | (3072, 18432) | 56,641,536 |
| `attn.a_to_qkv` | (3072, 9216) | 28,320,768 |
| `attn.b_to_qkv` | (3072, 9216) | 28,320,768 |
| `attn.a_to_out` | (3072, 3072) | 9,440,256 |
| `attn.b_to_out` | (3072, 3072) | 9,440,256 |
| `attn.norm_q_a` | RMSNorm(128) | 128 |
| `attn.norm_k_a` | RMSNorm(128) | 128 |
| `attn.norm_q_b` | RMSNorm(128) | 128 |
| `attn.norm_k_b` | RMSNorm(128) | 128 |
| `ff_a.0` | (3072, 12288) | 37,761,024 |
| `ff_a.2` | (12288, 3072) | 37,751,808 |
| `ff_b.0` | (3072, 12288) | 37,761,024 |
| `ff_b.2` | (12288, 3072) | 37,751,808 |
| | **Per joint block** | **339,831,296** |

Note: `norm2_a`, `norm2_b` are `LayerNorm(elementwise_affine=False)` — 0 trainable params.

### Per single block (full weights)

| Module | Linear dims | Params |
|--------|------------|------:|
| `norm.linear` | (3072, 9216) | 28,320,768 |
| `to_qkv_mlp` | (3072, 21504) | 66,081,792 |
| `proj_out` | (15360, 3072) | 47,188,992 |
| `norm_q_a` | RMSNorm(128) | 128 |
| `norm_k_a` | RMSNorm(128) | 128 |
| | **Per single block** | **141,591,808** |

### ControlNet-specific layers

| Module | Architecture | Params |
|--------|-------------|------:|
| `controlnet_blocks` | 5 × Linear(3072, 3072) | 47,201,280 |
| `controlnet_single_blocks` | 10 × Linear(3072, 3072) | 94,402,560 |
| `time_embedder` | Sequential(Linear(256,3072), SiLU, Linear(3072,3072)) | 10,229,760 |
| `guidance_embedder` | Sequential(Linear(256,3072), SiLU, Linear(3072,3072)) | 10,229,760 |
| `pooled_text_embedder` | Sequential(Linear(768,3072), SiLU, Linear(3072,3072)) | 11,802,624 |
| `context_embedder` | Linear(4096, 3072) | 12,585,984 |
| `x_embedder` | Linear(64, 3072) | 199,680 |
| `controlnet_x_embedder` | Linear(64, 3072) | 199,680 |
| | **CN-specific total** | **186,851,328** |

Note: `pos_embedder` (RoPEEmbedding) has 0 trainable parameters (precomputed sin/cos).

### Full ControlNet total

```
5  joint blocks:  5 × 339,831,296  = 1,699,156,480
10 single blocks: 10 × 141,591,808 = 1,415,918,080
CN-specific layers:                 =   186,851,328
                                    ──────────────────
Full ControlNet                     = 3,301,925,888
```

### Verified against model

```python
cn = FluxControlNet()
total = sum(p.numel() for p in cn.parameters())
# total == 3,301,925,888 ✓
```

---

## 9. Final Comparison

| Configuration | Params | % of Full CN | % of Full DiT |
|--------------|------:|:---:|:---:|
| **CN LoRA (rank 32)** | **40.3M** | 1.22% | 0.34% |
| Hypothetical DiT LoRA (rank 32) | 153.2M | 4.64% | 1.28% |
| Full ControlNet | 3,301.9M | 100% | 27.5% |
| Full FLUX.1-dev DiT | ~12,000M | ~363% | 100% |

### Why ControlNet LoRA and not DiT LoRA?

1. **3.8x fewer parameters** for the same rank — ControlNet has 15 blocks vs DiT's 57
2. **Preserves generative prior** — DiT weights stay frozen, retaining FLUX.1-dev's image quality
3. **ControlNet is the conditioning pathway** — it's architecturally designed to inject new conditioning signals (like SPAD data), making it the natural target for adaptation
4. **Lower VRAM** — 40.3M trainable params vs 153.2M means lower optimizer state memory (Adam stores 2 additional copies of trainable params)

### VRAM impact (Adam optimizer)

Adam stores m (first moment) and v (second moment) for each trainable parameter:

| | Trainable | Optimizer state (fp32) | Total training overhead |
|-|----------:|----------------------:|------------------------:|
| CN LoRA | 40.3M × 2B = 80.6MB | 40.3M × 8B = 322.4MB | ~403MB |
| DiT LoRA | 153.2M × 2B = 306.3MB | 153.2M × 8B = 1,225.3MB | ~1,531MB |
| Full CN | 3,301.9M × 2B = 6,604MB | 3,301.9M × 8B = 26,415MB | ~33,019MB |

---

## Appendix A: Verification Script

All parameter counts were verified by instantiating actual PyTorch modules:

```python
from diffsynth.models.flux_dit import FluxJointTransformerBlock, FluxSingleTransformerBlock
from diffsynth.models.flux_controlnet import FluxControlNet

jb = FluxJointTransformerBlock(3072, 24)
assert sum(p.numel() for p in jb.parameters()) == 339_831_296

sb = FluxSingleTransformerBlock(3072, 24)
assert sum(p.numel() for p in sb.parameters()) == 141_591_808

cn = FluxControlNet()
assert sum(p.numel() for p in cn.parameters()) == 3_301_925_888
```

All three assertions pass.

---

## Appendix B: Source File References

| File | What was used |
|------|--------------|
| `diffsynth/models/flux_dit.py:108-148` | FluxJointTransformerBlock definition |
| `diffsynth/models/flux_dit.py:205-258` | FluxSingleTransformerBlock definition |
| `diffsynth/models/flux_dit.py:40-104` | FluxJointAttention definition |
| `diffsynth/models/flux_dit.py:152-185` | FluxSingleAttention definition |
| `diffsynth/models/flux_dit.py:189-201` | AdaLayerNormSingle definition |
| `diffsynth/models/general_modules.py:116-128` | AdaLayerNorm definition (dim × 6 for default) |
| `diffsynth/models/general_modules.py:80-89` | TimestepEmbeddings definition |
| `diffsynth/models/flux_controlnet.py:61-79` | FluxControlNet.__init__ (5 joint + 10 single) |
| `train_scene_aware_raw.sh` | LoRA rank=32, target modules, base_model=controlnet |
