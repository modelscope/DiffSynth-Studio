# ConvRot 通用量化后端调研报告

- 日期：2026-08-04（v2，按三点确认要求重写）
- 目标：为 DiffSynth-Studio 量化框架（`diffsynth/core/quant`）新增一个**通用 ConvRot 后端**，覆盖 int8 / int4（及后续可扩展的 scheme），首个验证对象为 MiniMax-H3 DiT
- 验证素材：`models/Gluttony10/MiniMax-H3-INT8-CONVROT/`（本地已下载）
- 探测脚本：`workspace/20260804_convrot_quant/`，本报告所有数字均可复现
- 本报告只做调研与方案设计，**未修改任何仓库代码**

## v2 相对 v1 的三点变更（对应你的要求）

| 要求 | 落地结论 |
| --- | --- |
| ① 自定义 Linear 必须继承 `nn.Linear` | 采纳并定死。`ConvRotLinear(nn.Linear)`，实现细节见 §7.5。这与 `bnb.nn.Linear4bit`（本身就是 `nn.Linear` 子类）一致，还顺带让 `auto_detect_lora_target_modules` 与 peft 能识别量化层——`Fp8Linear`（纯 `nn.Module`）识别不到。 |
| ② ConvRot 做成通用后端，多个注册条目 | 采纳。调研确认 ComfyUI 的 ConvRot 是**两个** `QUANT_ALGOS` 格式（`int8_tensorwise` 的 convrot 选项 + 独立的 `convrot_w4a4`），且两者共用同一套 `weight`/`weight_scale`/`comfy_quant` 存储契约。方案：**一个 backend + 可插拔 scheme 表 + 4 个注册条目**（§7）。 |
| ③ `_disk_required_keys()` 上升为框架接口 | 采纳。新增 `QuantBackend.checkpoint_key_patterns()` + `QuantizeConfig/MixedQuantizeConfig.checkpoint_keys()`，`core/vram/layers.py` 只调用 config 层（顺带修好 `MixedQuantizeConfig` + disk offload 目前直接崩的 `.backend` 反向依赖）。完整设计见 §8。 |

---

## 1. 结论摘要

1. **ConvRot 有正式论文，也有事实上的官方实现，但论文自己的代码没公开**（§2）。三层要分清：算法定义来自 arXiv 2512.03673（清华 + 华为）；论文官方仓库**遍寻未获**；事实标准实现是 **Comfy Org 官方的 `comfy-kitchen`（Apache-2.0，PyPI 可装）+ ComfyUI**，且它与论文方法**完全一致**。
2. **格式全部查清，零黑盒。** 权威来源四处：论文、ComfyUI `comfy/quant_ops.py` 的 `QUANT_ALGOS`、`comfy/ops.py` 的 marker 读写实现、`comfy-kitchen` 的 layout 与 kernel。
3. **ConvRot 家族 = 2 个格式**：`int8_tensorwise`（W8A8，可选 convrot）与 `convrot_w4a4`（W4A4，convrot 内建、int4 打包）。两者 marker 与边料键名同构，**共用一个 backend 是正确的抽象边界**。
4. **数学已被官方 kernel 逐元素交叉验证**：我的手搓逆旋转与 `torch.ops.comfy_kitchen.dequantize_int8_convrot_weight` **max|diff| = 0.0**；W8A8 前向与官方 `int8_linear` 也 **max|diff| = 0.0**。反量化还原 vs 官方 bf16 权重：余弦 **0.999962**、相对 L2 **0.0087**。
5. **⚠️ 上一版的「零新增依赖」判断被实测推翻，但结局更好**（§6.2、§6.3）：comfy-kitchen 的 **triton** 路径在本机 1.55–1.58×，而我们的 eager 只有 0.98×。但两边都是 **Apache-2.0**且它的 triton 路径**全是纯 Python**（`.so` 只属于 cuda/hip 后端）—— 已实测**移植 ~215 行到我们自己文件里：6.40 ms / 1.59×，与上游输出 max|diff| = 0**。⇒ 策略：**移植 kernel，不依赖包**；它的 CUDA kernel 需 cu13，本机（torch cu128 / driver 570.133.20）跑不起来，ComfyUI 自己也会在 cuda<13 时禁用。
6. **W4A4 的结论要重新表述**（§5.3）：我测的 per-row 粒度**正是论文的粒度**（per-token/per-channel），不是 comfy 的退化实现；而论文自己也承认纯 W4A4 有可接受退化（FID 12.32 vs BF16 10.07），它给出的答案是**混合精度：20% 最敏感层 INT8 + 其余 INT4**（FID 10.03）——这恰好就是 DiffSynth 的 `MixedQuantizeConfig`。
7. **需要你确认的 core 改动共 3 处**（§8.4），其中 1 处必需、2 处为顺手加固。

---

## 2. ConvRot 的来源、官方实现与格式全景

### 2.1 算法出处：论文（权威定义）

**ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers**，arXiv:2512.03673，2025-12-03，Feice Huang / Zuliang Han / Xing Zhou / Yihuang Chen / Lifei Zhu / Haoqian Wang（清华，署名单位含华为）。

论文要点（与我们的实现直接相关的部分）：

- **为什么必须是 regular Hadamard（RHT），不能是 Sylvester 型**：DiT 里存在**逐行离群值**（LLM 里通常是逐列）。Sylvester 型 Hadamard（也就是 FWHT / `scipy.linalg.hadamard` 用的那种）**第一列全为 1**，作用在逐行离群值上会把离群能量**集中而非打散**。论文定义列差异 `max_j |Σ_i H_ij|`，证明「每行每列和均为 ±√n」的 regular Hadamard 取到最小可能列差异 √n。⇒ **这是实现里最危险的一处：谁"顺手"把 kron(H4) 换成 `scipy.linalg.hadamard`，数值会变差且与 checkpoint 不兼容。必须在代码注释里写死。**
- **Kronecker 构造**支持任意 `4^k` 块大小；消融显示 **N₀ = 256 是效率/精度最优点**（正是发布 checkpoint 用的值）。
- **不用 FWHT 蝶形，改用 reshape + matmul**，以吃现代 GPU 的 GEMM 流水线（我们的实现也是这么做的，与论文一致）。
- **分组把复杂度从 O(K²) 降到 O(K)**；DiT 的 AdaLN 破坏了 LLM 里"把旋转融进前一层"的技巧，所以在线旋转无法省掉。
- **ConvLinear4bit** = 把「旋转 → 量化 → 4bit GEMM → 反量化」封装成可直接替换 `nn.Linear` 的模块。这与我们 `ConvRotLinear` 的设计是同一个思路。
- 量化粒度：**per-token（激活）/ per-channel（权重）**。
- FLUX.1-dev/schnell (12B) 结果：内存 22.7 → 5.6 GiB（4.05×）、2.26× 加速；纯 INT4 FID 12.32，**混合精度（20% 层 INT8）FID 10.03**，BF16 基线 10.07。
- 论文明确说 kernel 建立在 **QuaRot** 之上，主要对标 **SVDQuant**（后者需要专用推理引擎，ConvRot 卖点是即插即用）。

### 2.2 官方实现现状（三层，务必分清）

| 层 | 状态 | 我们能不能用 |
| --- | --- | --- |
| 论文作者的官方代码 | **未找到公开仓库**：arXiv abs 页无 code 链接，PapersWithCode / GitHub / 中文社区多路检索均无结果。有 OpenReview 记录（`SCC11m676G`）但被人机验证挡住 | 无 |
| **`comfy-kitchen`（Comfy Org 官方，Apache-2.0，PyPI）** | **这就是事实标准实现**。0.2.26 提供 45 个算子，含 `int8_linear`、`quantize/dequantize_int8_convrot_weight`、`convrot_w4a4_linear`、`quantize_and_rotate_rowwise` 等；三套后端 eager / triton / cuda。其 `_build_hadamard`（kron(H4) / 4 的幂 / 除 √size）、reshape-matmul 旋转、group 256 与论文**逐条对应** | **可用**：cp310 wheel 在本机 import 正常，**triton 路径可用且快**（§6.2）；cuda 路径需 cu13，本机不可用 |
| ComfyUI 本体（`comfy/quant_ops.py` + `comfy/ops.py`） | 定义 checkpoint 格式契约（`QUANT_ALGOS` + marker 读写），是「哪些文件能被谁读」的唯一权威 | 我们照它实现存储契约 |

⇒ **对你的问题的直接回答：算法有权威论文；论文官方代码没公开；但有 Comfy Org 官方维护的实现（comfy-kitchen），而且我们已经实测它在本机的可用性与性能，并用它交叉验证了我们的手搓数学（逐元素为 0 差异）。**

### 2.3 生态调研：其他专业量化后端支持 ConvRot 吗？

**结论：没有。一个都没有。** 旋转类量化在工业界确实成熟，但没有任何主流后端支持 ConvRot —— **格式不通，而且旋转矩阵本身就不是同一个**。

| 后端 | 旋转能力 | 用的是什么 Hadamard | 能读 convrot checkpoint 吗 |
| --- | --- | --- | --- |
| **torchao** `prototype/spinquant` | 有（SpinQuant/QuaRot 移植） | `hadamard_transform` docstring 明写等价于 `scipy.linalg.hadamard` ⇒ **Sylvester**；`random_hadamard_matrix` = `diag(±1) @ Sylvester`（QuIP# 式）；依赖 Dao-AILab `fast-hadamard-transform` 的 FWHT kernel | ❌ |
| **llm-compressor / compressed-tensors**（vLLM 生态） | 有（0.7.0 起的 transforms）：`SpinQuantModifier`、`QuIPModifier` | 同样 Sylvester / 随机 Hadamard | ❌（另有自己的 compressed-tensors 格式） |
| **QuaRot** 官方仓库 | 有 | Sylvester + 已知 Hadamard 块（12/20/28/…）。ConvRot 论文明言 kernel 建在它之上，**但换了矩阵** | ❌ |
| **SVDQuant / nunchaku**（扩散模型 W4A4，最接近的对手） | **不用旋转** | —（用 smoothing + 低秩分支吸收离群值） | ❌（ConvRot 论文正是拿它当主要对标，卖点是不需专用引擎） |
| **TensorRT Model Optimizer** | FP8/FP4/SVDQuant | — | ❌ |
| **bitsandbytes** | 无 | — | ❌ |

**实测决定性证据**（`probe_hadamard_variants.py`，真实权重 `blocks.0.mlp.fc1`）：

| 256×256 变换 | 正交 | 对称 | 列差异（理论最小 = √256 = 16） | 能否反旋转发布的 checkpoint |
| --- | --- | --- | --- | --- |
| **regular kron(H4)**（ConvRot） | ✓ | ✓ | **16.0**（取到理论最小） | ✅ cosine **+0.999962** |
| sylvester kron(H2)（FWHT / torchao） | ✓ | ✓ | **256.0**（最差，第一列全 1） | ❌ cosine **+0.0037**，rel_l2 1.41 ⇒ 完全是噪声 |
| random diag(±1)@H2（QuIP#/SpinQuant） | ✓ | ✗ | 60.0 | ❌ cosine **−0.0007**，rel_l2 1.41 |

✅ 论文的列差异理论在数值上完全成立；✅ **现成库的 Hadamard 不能替换，格式硬绑定 regular Hadamard**。

**但要诚实说一个反向发现**：在**权重** per-row 量化误差上，三种旋转几乎无差别（int8：0.00882 / 0.00883 / 0.00883；int4：0.16009 / 0.16011 / 0.16014），且旋转相对不旋转的收益也很小（int8 0.01022 → 0.00882，仅 14%）。⇒ regular 的优势应该体现在**激活**侧（DiT 的逐行离群值在激活上），我这个实验只测了权重，**不构成对论文消融的复现**（要验证得抓真实激活分布）。这不影响结论（格式绑定），但我不能声称"regular 更准"已在本模型上得到证实。

**能复用的与不能复用的**：

| | 能不能用 |
| --- | --- |
| 旋转层（Hadamard） | ❌ 不能。矩阵不同，换了就读不了 checkpoint。必须自己写 kron(H4)（十几行代码） |
| 存储层（marker / 边料键） | ❌ 不能。ComfyUI 私有格式，torchao / bnb / compressed-tensors 三家各自一套 |
| **GEMM 层** | ✅ **可以不手搓**：`torch._int_mm`（cuBLASLt IMMA）+ triton 写融合尾巴。关键事实：**triton 是 torch 的硬依赖**（`triton==3.6.0; platform_system == "Linux"`），写 triton kernel **不算引入新依赖** |
| 方法论 | ✅ 论文的混合精度（20% INT8 + 80% INT4）→ 直接映到我们的 `MixedQuantizeConfig`；敏感层分析可借 llm-compressor 的思路 |

⇒ **自研 backend 是唯一选项（格式独有），但并不意味着所有东西都要手搓 —— 旋转与 marker 必须自己写（很短），GEMM 可以交给 `torch._int_mm` + triton（零新依赖）。**

### 2.4 格式全景

ComfyUI `comfy/quant_ops.py::QUANT_ALGOS` 全表（master，2026-08）：

| format | 权重 dtype | 边料键 | group | ConvRot | layout 类 |
| --- | --- | --- | --- | --- | --- |
| `float8_e4m3fn` / `float8_e5m2` | fp8 | `weight_scale`, `input_scale` | — | 无 | `TensorCoreFP8*Layout` |
| `mxfp8` | fp8_e4m3 | `weight_scale`（按 `float8_e8m0fnu` view） | 32 | 无 | `TensorCoreMXFP8Layout` |
| `nvfp4` | uint8 | `weight_scale`(fp8_e4m3 块) + `weight_scale_2` + `input_scale` + `pre_quant_scale` | 16 | 无 | `TensorCoreNVFP4Layout` |
| **`int8_tensorwise`** | **int8** | **`weight_scale`** | — | **可选**（`convrot` + `convrot_groupsize`，默认 256） | `TensorWiseINT8Layout` |
| **`convrot_w4a4`** | **int8（打包 2 个 nibble）** | **`weight_scale`** | **quant 64 / convrot 256** | **内建** | `TensorCoreConvRotW4A4Layout` |

另外 comfy-kitchen 里还有 `TensorCoreAWQW4A16Layout` 与 `TensorCoreSVDQuantW4A4Layout`，但**没有**进 ComfyUI 的 `QUANT_ALGOS`，也不走 convrot（SVDQuant 用低秩补偿分支、AWQ 用逐通道 smoothing），因此**不在本次 backend 范围内**（结构上可以后续作为新 scheme 加入，见 §7.4 的扩展位）。

ConvRot 算法本体（`comfy_kitchen/tensor/int8_utils.py` 与 `backends/eager/convrot_w4a4.py` 两份实现完全一致）：

```
H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
H  = kron(H4, H4, ...) / sqrt(size)     # size 必须是 4 的幂：4/16/64/256/1024...
```

`H` **对称且正交**（`H @ H = I`）⇒ 旋转与逆旋转是同一个函数，这是实现里最省事也最容易写错的一点（实测已验证）。

- 权重（离线）：`W_rot = W.view(out, in//gs, gs) @ H`
- 激活（在线）：`x_rot = x.view(-1, in//gs, gs) @ H`
- 因为 `(x@H) @ (W@H)^T = x @ W^T`，旋转不改变数学，只改变量化前的数值分布。

---

## 3. 存储契约与 marker schema（完整）

### 3.1 每个量化层的键

```
<layer>.weight        低比特权重（int8 / 打包 int4）
<layer>.weight_scale  float32
<layer>.comfy_quant   uint8，json.dumps(...) 的原始字节
<layer>.bias          原 dtype，不量化
```

### 3.2 marker JSON schema（`comfy/ops.py` 读写实现的精确复刻）

```jsonc
{
  "format": "int8_tensorwise",          // 必需；不在 QUANT_ALGOS 里 → ValueError
  "convrot": true,                      // int8_tensorwise 专有，可选
  "convrot_groupsize": 256,             // 同上，默认 256
  "linear_dtype": "int8",               // convrot_w4a4 专有，默认 "int4"，等于默认时不写
  "full_precision_matrix_mult": true    // 可选；置位则强制走反量化 matmul
}
```

写入端（`_quantized_weight_state_dict`）的行为，我们的 `flatten_state_dict` 必须逐条对齐：

- `format` 永远第一个；
- `int8_tensorwise` 仅当该层**真的旋转过**才写 `convrot` / `convrot_groupsize`（`in_features % 256 != 0` 的层不旋转，marker 退化为 `{"format": "int8_tensorwise"}`）；
- `convrot_w4a4` 总写 `convrot_groupsize`，`linear_dtype` 仅在 ≠ `"int4"` 时写；
- `full_precision_matrix_mult` 仅在配置置位时写（comfy-quants 的导出器刻意从不写它）。

读取端（`_load_quantized_weight`）有一个**兼容分支必须实现**：convrot 字段可能藏在嵌套的 `"params": {...}` 里，读的时候要 `layer_conf.get("convrot", params_conf.get("convrot", False))`。我们的解析器要同时吃扁平与嵌套两种形态。

`convrot_w4a4` 的 `quant_group_size` 在 ComfyUI 读取端是**硬编码 64**（不从 marker 读）。

### 3.3 与已废弃 `int8_w8a8` 的区分

ComfyUI-INT8-Fast 的旧格式 marker 是 `{"convrot": …, "per_row": true}`，**没有 `format` 字段**，且量化数学用 fp32 倒数乘而非除法。⇒ 我们的解析器遇到缺 `format` 的 marker 必须**报错而不是猜**。

---

## 4. 目标 checkpoint 实测事实

### 4.1 model hash 与注册现状

| 文件 | 大小 | `hash_model_file` | 现状 |
| --- | --- | --- | --- |
| `MiniMax-H3-FL2VA-int8_convrot.safetensors` | 43.8 GiB | `68f08b5dd411f3798fc73f4699bb1d0e` | 待注册 |
| `MiniMax-H3-Ref2VA-int8_convrot.safetensors` | 43.8 GiB | 同上（**共用 hash**） | 一条注册即可覆盖两者 |
| `qwen3-vl-32b-int8_convrot.safetensors` | 25.3 GiB | `7b7cf3198d4a0522bf8892f1adcc63e1` | 第二阶段 |
| `MiniMax-H3-video_vae.safetensors` | 9.7 GiB | `24b80900992e2024fab17c991c57da23` | **已注册**（就是现有 bf16 VAE 的单文件版，零工作量） |
| `MiniMax-H3-audio_vae.safetensors` | 0.56 GiB | `db383f1c8960837b94059f7722e6cb11` | **已注册**，同上 |

### 4.2 DiT 结构（937 张量）与键名

**键名与 `MiniMaxH3DiT` 完全一致，不需要 state_dict_converter。**

| 部分 | 处理 | 体积 |
| --- | --- | --- |
| 201 个 Linear：`blocks.{0..49}.{attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2}` + `condition_proj` | INT8 + ConvRot | 17.97 GiB |
| `blocks.*.adaln_proj.linear` (`[96768,2688]`×50) + `final_layer.adaln_proj.linear` | BF16（adaLN 从不量化） | 占 25.72 GiB 的绝大部分 |
| `token_refiner.blocks.{0,1}.*`（8 个 Linear） | BF16 | 同上 |
| `video/audio_patch_proj`、`time_embedder.*`、`final_layer.{video,audio}_out`、`rope.inv_freq` | FP32 | 0.08 GiB |

- 201 层 marker **字节级完全一致**：`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`
- 201 层 `in_features` ∈ {5376, 7168, 14336, 5120}，**全部被 256 整除** ⇒ 全部真旋转，无退化层
- 43.8 GiB 里 58% 是未量化的 adaLN；上游另有 `adaln_curve` 变体可压到 ~21 GiB，属模型结构改动，本次不做

### 4.3 文本编码器（第二阶段）

- 键名沿用 HF `model.language_model.*` / `model.visual.*`，现有 `MiniMaxH3TextEncoderStateDictConverter` 直接可用
- 350 个量化层 = 50 层 × 7 个 Linear（`q/k/v/o_proj`, `gate/up/down_proj`），**visual tower 完全不量化**（0 marker）⇒ 目标层可用后缀匹配干净表达

---

## 5. 数值验证

### 5.1 int8 反量化对齐官方 bf16（✅）

| 层 | shape | 余弦 | 相对 L2 |
| --- | --- | --- | --- |
| `blocks.0.mlp.fc1` | (28672, 5376) | 0.999962 | 0.008667 |
| `blocks.0.attn.qkv_proj` | (21504, 5376) | 0.999962 | 0.008679 |
| `blocks.25.attn.out_proj` | (5376, 7168) | 0.999954 | 0.009606 |
| `condition_proj` | (5376, 5120) | 0.999963 | 0.008641 |

### 5.2 无法 bit 级复现该第三方 artifact（❌，但不影响加载）

按文档的 absmax 配方重新量化官方权重，只有 ~48% 码字一致（最大差 8）。定位结果：

- 发布 scale 比「fp32 旋转后 absmax/127」**系统性偏小**，比值 `[0.937, 1.000]`、均值 0.968，**从不超过 1.0**
- 每行 `max|code|` 恰好都是 127（峰值被 clip 到边界）
- 用发布的 scale 反算码字，与发布码字 **99.99% 一致、最大差 1**

⇒ 生产方（RunningHub 插件的 `tools/quantize_int8_convrot.py`）额外做了一步 **clipping 搜索**（MSE 最优裁剪，系数约 0.94–1.00），不是 comfy-quants 文档的朴素配方。只影响「我们自己产 checkpoint」，不影响加载。

**官方 kernel 的交叉验证（新增，很干净）**：直接调 `torch.ops.comfy_kitchen.quantize_int8_convrot_weight(w, 256)` 重新量化官方 bf16 权重，得到的码字与发布文件一致率 **0.486362**、max|dq| = 8、scales 不等 —— **与我手搓配方的数字一位不差**（同层同样是 0.486362）。这同时证明两件事：

1. 我的量化配方与官方 kernel **逐位一致**（不是"差不多"）；
2. 该第三方 checkpoint 确实**不是**用官方朴素配方产出的，对方多做了一步裁剪搜索。

### 5.2b 反量化路径被官方 kernel 逐元素确认（✅）

| 对比 | 结果 |
| --- | --- |
| 我的逆旋转 vs `dequantize_int8_convrot_weight` | **max|diff| = 0.000e+00** |
| 我的 W8A8 前向 vs `int8_linear`（convrot=True, gs=256） | **max|diff| = 0.0000**，rel_l2 同为 0.01304 |

⇒ §5.1 的数学结论不再只是"我算的"，而是与 Comfy Org 官方实现比特级一致。

### 5.3 W4A4 精度实测（⚠️ 差 —— 但这与论文一致，不是实现问题）

同一层 `blocks.0.mlp.fc1`，ConvRot + 对称 int4（emission range `[-7,7]`，scale = absmax/7）：

| 方案 | 权重 rel L2 | 端到端 linear rel L2 |
| --- | --- | --- |
| int4 per-row（= comfy-kitchen eager 的做法） | 0.160 | 0.228 |
| int4 per-group-64 | 0.108 | — |
| int4 per-group-32 | 0.097 | — |
| int8 per-row | 0.0095 | 0.0136 |

打包编解码自洽性：`pack/unpack` 往返**完全无损**，`[28672,5376] → [28672,2688] int8`，0.50 bytes/param（✅ 规格正确）。

**重要更正（v1 表述有误导）**：per-row 不是 comfy 的"退化实现"，**它就是论文的粒度**（per-token 激活 / per-channel 权重）。而论文自己的数据也说纯 W4A4 只是"可接受退化"（FLUX FID 12.32 vs BF16 10.07），它给出的解法是**混合精度：20% 最敏感层用 INT8，其余 INT4**（FID 10.03，基本追平基线）。

⇒ 结论应该这样写：**ConvRot W4A4 单独用不行，论文原本就没打算单独用。** 而"20% 层 INT8 + 80% 层 INT4"在 DiffSynth 里天然就是 `MixedQuantizeConfig(configs=[QuantizeConfig(method="comfy_convrot_w4a4", ...), QuantizeConfig(method="comfy_int8_convrot", target_modules=敏感层)])` —— 我们的框架恰好已经有这个能力，这反而是接入 ConvRot 的一个**结构性优势**（敏感层清单需要按论文的方法做逐层敏感度分析，属阶段 3）。

---

## 6. 运行时实测与选型（A100-80GB, torch 2.10.0+cu128, tokens=8192）

### 6.1 手搓 PyTorch 路径

| 层 | bf16 | W8A16（权重预还原） | W8A8（`_int_mm`） | W8A8 误差 | W8A16 误差 |
| --- | --- | --- | --- | --- | --- |
| `attn.qkv_proj` | 7.99 ms | 7.40 ms | 8.23 ms | 0.0130 | 0.0091 |
| `mlp.fc1` | 9.89 ms | 9.93 ms | 10.58 ms | 0.0130 | 0.0091 |
| `mlp.fc2` | 4.82 ms | 5.00 ms | 6.16 ms | 0.0137 | 0.0095 |

W8A8 拆解（`mlp.fc1`）：`_int_mm` **6.42 ms** + 激活旋转量化 0.95 ms + fp32 反缩放 3.21 ms。

**三个必须写进实现的硬约束：**

1. **`_int_mm` 的 B 布局决定生死。** 传 `q.T`（column-major 视图）= 6.42 ms；传 `q.T.contiguous()`（row-major）= **35.17 ms**，比 bf16 慢 3.5×。comfy-kitchen 的 eager 兜底恰好用了 `.contiguous()` —— **绝对不要照抄它的 eager 实现**，它靠 CUTLASS 内核吃饭。
2. **`_int_mm` 要求 `M > 16`**（实测 M=16 报错，M=17 通过），K/N 需 8 对齐。`condition_proj` 吃文本 token，短 prompt 下 M 可能 <17 ⇒ **必须做行 padding**（pad 到 `max(M,32)` 向上取 32 倍数，算完裁回）。
3. **eager 的 fp32 反缩放尾巴是瓶颈**（int32 `[M,N]` → fp32 → bf16，纯带宽）。`torch.compile(dynamic=False)` 能融合：同 shape 下 bf16 10.48 / W8A8 eager 10.45 / **W8A8 compiled 7.51 ms（1.4×）**。

另两条：

- **动态 W8A16**（显存里保持 int8，每次 forward 现场反量化）：反量化 1.81 ms（bf16 数学）/ 6.26 ms（fp32 数学），端到端 11.41 ms vs bf16 10.42 ms，**只贵 10%**；bf16 数学做逆旋转额外引入 0.32% 权重误差（相对 int8 本身 0.9% 可忽略）。这条路径**可微**，是 LoRA 训练与非 CUDA 设备的天然回退。
- **A100 上 fp8 没戏**：`torch._scaled_mm` 要求 SM ≥ 8.9，A100(8.0) 直接报错。int8 是 A100 上唯一可用的 8-bit tensor-core 路径。

**选型（已按 §6.2 的实测修正）**：默认 W8A8（这也是 ComfyUI 的真实语义），不可用时回退反量化路径；**检测到 comfy-kitchen 时优先用它的 kernel**（1.55×），否则用手搓路径（约持平，可选 `torch.compile` 到 1.4×）。**第一阶段承诺的收益是显存（201 层 36 → 18 GiB）；提速取决于是否装了 comfy-kitchen。**

### 6.2 官方 comfy-kitchen kernel 实测（推翻 v1 的「零依赖」判断）

把 0.2.26 的 cp310 wheel 解包到临时目录、只用 `PYTHONPATH` 挂载（**不污染 debug 环境**）实测：

| 路径 | 耗时 | 相对 bf16 | 数值 |
| --- | --- | --- | --- |
| bf16 `F.linear` | 9.79 ms | 1.00× | 基线 |
| 手搓 `torch._int_mm`（eager） | 11.72 ms | 0.84× | rel_l2 0.01304 |
| 手搓 + `torch.compile` | 7.51 ms | 1.30× | 同上 |
| **comfy-kitchen `int8_linear`（triton）** | **6.34 ms** | **1.55×** | rel_l2 0.01304，**与手搓 max|diff| = 0.0** |
| comfy-kitchen `int8_linear`（cuda） | ❌ | — | `CUDA driver version is insufficient for CUDA runtime version`（shared memory 54272B 的 convrot64 fused kernel） |

关键事实：

- **wheel 本身在 torch 2.10.0+cu128 下 import 正常**，`ck.list_backends()` 报 eager / triton / cuda 三套均 available —— 但 **cuda 那套是假可用**：真正调用时报驱动版本不足。它的 CUDA kernel 是按 **cu13** 编的，ComfyUI 自己在 `quant_ops.py` 里就有 `if cuda_version < (13,): ck.registry.disable("cuda")` 的门禁。本机 driver 570.133.20 / torch cu128 ⇒ 只能走 triton。
- **triton 路径又快又对**：比手搓 eager 快 1.85×，比手搓 `torch.compile` 还快 16%，且输出与手搓**逐元素完全相同**。
- 官方 op 的签名与我 v1 的假设有出入（`quantize_int8_convrot_weight(Tensor weight, SymInt group_size)` 只收 2 个参数，没有 `stochastic_rounding`）—— 说明**不能照 Python 源码的签名猜 `torch.ops` 的签名**，要以实际注册为准。

**因此依赖策略改为：**

| | 方案 |
| --- | --- |
| 默认 | 自实现（`torch._int_mm` + 手搓 RHT + 位打包）。零依赖、任何环境可跑、可微、能做 `dequant_once`。**正确性的唯一来源。** |
| **推荐的加速方式** | **把上游的两个 triton kernel 移植进来（同为 Apache-2.0）** —— 已实测与上游**逐元素相同、性能持平**，且零依赖。详见 §6.3 |
| 不做 | 把 comfy-kitchen 设为硬依赖（运行时 import）。它的加速依赖 triton/cu13 版本组合，且 `pyproject.toml` 里多一个带预编译 CUDA 扩展（135 MB 的 `.so`）的依赖，对整仓库的安装成功率影响太大 |

---

### 6.3 直接移植上游 kernel（推荐方案，已实测）

#### 实测结果：完美复现

把上游的两个 triton kernel 抄到我们自己的文件里（`probe_ported_kernels.py`），同一 shape：

| 路径 | 耗时 | 相对 bf16 | 与上游的数值差异 |
| --- | --- | --- | --- |
| bf16 `F.linear` | 10.17 ms | 1.00× | — |
| 我们的 eager | 10.43 ms | 0.98× | — |
| comfy-kitchen 上游（triton） | 6.46 ms | 1.58× | — |
| **移植到我们文件里** | **6.40 ms** | **1.59×** | **max\|diff\| = 0**（逐元素相同） |

另一个收获：搞清了性能差到底在哪里 —— **不在 GEMM，在激活量化。** 我的 eager 版用 torch 多算子（rotate → amax → div → round → clamp → to(int8)，反复读写 x）；上游用一个 triton kernel、一个 program 处理一整行、一次 pass 搞完。所以我自己写的完整 triton GEMM 只到 7.20 ms，而抄了他们的量化 kernel 后直接到 6.40 ms。

#### 许可：可以移植，但要带署名

| | |
| --- | --- |
| comfy-kitchen | `License: Apache-2.0`，dist-info 带 LICENSE + NOTICE |
| DiffSynth-Studio | Apache-2.0 |
| 结论 | 兼容，可移植 |

署名链比想象的复杂，三层都要写：

- `backends/triton/quantization.py` 的 SPDX 头是 **NVIDIA CORPORATION** 的版权（不是 Comfy Org）
- 包由 **Comfy Org** 发布（comfy-kitchen，Apache-2.0）
- INT8 tensorwise 那一节上游自己注明："from **dxqb/OneTrainer** & **ComfyUI-Flux2-INT8**"

Apache-2.0 §4 的义务：保留版权与许可声明、**显著地声明我们作了修改**。落地方式：移植文件头部原样保留 SPDX 头 + 一段 provenance 注释，列出上游路径、版本与我们的改动。

#### 搜什么、不搜什么

| | 内容 | 行数 |
| --- | --- | --- |
| ✅ 搜 | `_quantize_rowwise_kernel`（融合的逐行量化） | ~45 |
| ✅ 搜 | `_int8_matmul_dequant_per_row_kernel`（per-channel scale 版，尾巴融合 dequant + bias） | ~65 |
| ✅ 搜 | `_int8_matmul_dequant_kernel`（标量 scale 版，兼容非 per-channel 的 checkpoint） | ~65 |
| ✅ 搜 | `int8_linear` 的调度逻辑（选 kernel / 算 strides / grid） | ~40 |
| ❌ 不搜 | `registry.py`（307 行的多后端优先级 + 约束系统） | — |
| ❌ 不搜 | `QuantizedTensor` + layout 体系（tensor subclass）—— 与我们的 `QuantBackend` 是两套并行架构，搜进来是异物 | — |
| ❌ 不搜 | `__init__.py`（819 行）、`constraints.py`、`float_utils.py` 的 fp8/fp4 部分 | — |
| ❌ 不需要 | 任何 `.so` —— cuda/hip 后端才是编译产物（135 MB / 36 MB），triton 与 eager **全是纯 Python** | — |

合计移植量 **~215 行纯 Python**。

#### 移植时不要"顺手优化"的四处细节

1. **`libdevice.rint`**：`from triton.language.extra import libdevice`。ComfyUI 自己的注释里就说过 "older Triton lacks `libdevice.rint` on the HIP backend and hard-crashes the INT8 path" ⇒ 需加 import 失败降级（退回 eager），不能裸用。
2. **`offs_am = (...) % m` 的取模 wrap**（而不是 mask）—— 靠 store 的 mask 兜底。改成 mask 会变性能特征，不要动。
3. **`tl.dot(a, b)` 不传 `out_dtype`**，靠 accumulator 的 int32 推断。
4. **B 的 strides**：`stride_bk=weight.stride(1)`（=1）、`stride_bn=weight.stride(0)`（=K）—— 即把 `[N,K]` 权重当`[K,N]` column-major 读。与 §6.1 独立发现的"必须 column-major"完全一致。

另一个实用发现：**旋转本身不是 triton kernel**，上游的 `int8_utils._rotate_activation` 就是 plain `torch.matmul` ⇒ 我们的旋转实现已经和它一样，无需改。

---

## 7. 后端方案（通用 ConvRot 后端，完整实现细节）

### 7.1 文件与命名

```
diffsynth/core/quant/backends/convrot.py      # 新增，~320 行
diffsynth/core/quant/backends/__init__.py     # + from . import convrot
```

backend 名 `"convrot"`。之所以放 core 而不是 model 目录：这套格式是 ComfyUI 生态通用的（Qwen-Image / LTX-2 / Wan / Flux 都有同格式发布），放 model 里以后必然到处复制。**属于共用框架代码，等你确认。**

### 7.2 旋转原语

```python
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

def build_hadamard(size, device, dtype):
    """归一化的 regular Hadamard，size 必须是 4 的幂。H 对称正交 ⇒ 正反旋转同一个函数。"""
    # 校验：size >= 4 且 log4(size) 为整数，否则 ValueError（与 comfy-kitchen 一致）
    # kron(H4) 累乘到 size，最后 / sqrt(size)；按 (size, device, dtype) 缓存

def rotate_last_dim(x, h, group_size):
    """x[..., K] -> reshape(-1, K//gs, gs) @ h -> 原 shape。K % gs != 0 时 ValueError。"""
```

实现要点：

- ⚠️ **必须是 regular Hadamard（kron(H4)），不能用 Sylvester 型**（`scipy.linalg.hadamard` / FWHT 用的那种，kron(H2)）。论文的核心贡献就在这里：Sylvester 型的第一列全为 1，作用在 DiT 的**逐行离群值**上会把离群能量集中、**放大**而非抑制（§2.1）。换错两个后果：精度变差 + 与现有 checkpoint 完全不兼容。**这句话要写在函数 docstring 里。**
- size 必须是 **4 的幂**（kron(H4) 的结构使然），不是 2 的幂；不满足直接 `ValueError`（与官方一致）
- 缓存 key 必须含 dtype 与 device，否则每层每次重建（256×256 的 kron 不贵但调用极频）
- 权重侧离线用 `h.T`、激活侧在线用 `h` —— 因为对称，二者等价；代码里**统一用 `h`** 并在注释里写明对称性，避免后人"修正"成 `h.T` 时以为改了语义
- 逆旋转的数学 dtype：**权重反量化必须走 fp32**（bf16 逆旋转会引入 0.32% 误差；只在"每次 forward 现场反量化"的热路径上才允许用 bf16 换速度，且要作为显式开关）

### 7.3 4-bit 打包编解码（`convrot_w4a4` 用）

```python
def pack_int4_row_major(values):    # (..., K) int8 -> (..., K//2) int8，低 nibble = 偶数列
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)

def unpack_int4_row_major(packed):  # 有符号 nibble 解释：>= 8 的值减 16
```

- K 必须为偶数
- 量化 emission range 是 **`[-7, 7]`（`_INT4_MAX = 7`，scale = absmax/7）**，不是 `[-8, 7]`：nibble 能表示 -8 但生产端刻意不发射，以保持 W4A4 kernel 契约的对称性。**写错成 8 会与 ComfyUI 的数值不一致。**
- 解码端仍要能吃满 `[-8, 7]`（存储级 codec 要容忍任何落进 nibble 的位模式）

### 7.4 Scheme 抽象（通用性的落点）

```python
class ConvRotScheme:
    marker_format: str            # comfy_quant["format"] 的值
    storage_dtype: torch.dtype    # <layer>.weight 的 dtype
    convrot_default: bool         # int8: 由配置决定；w4a4: 恒 True
    quant_max: int                # 127 / 7

    def storage_shape(out_f, in_f) -> tuple            # int8: (out,in)   w4a4: (out, in//2)
    def scale_shape(out_f, in_f) -> tuple              # int8: (out,1)    w4a4: (out,)
    def quantize_rows(rotated) -> (qdata, scale)       # 逐行 absmax 对称量化（+ 可选打包）
    def to_int8_operand(qdata) -> Tensor               # 给 _int_mm 用的 int8 视图（w4a4 = unpack）
    def dequantize_rows(qdata, scale, dtype) -> Tensor # 仍在旋转基下的浮点权重
    def quantize_activation(x_rot) -> (x_int8, x_scale)# 量化到本 scheme 的 range，**始终保持 int8 容器**
    def marker_payload(state) -> dict                  # 写：见 §3.2 的逐条规则
    def parse_marker(marker) -> dict                   # 读：扁平 + 嵌套 params 两种形态
```

两个内置 scheme：

| | `Int8TensorWiseScheme` | `ConvRotW4A4Scheme` |
| --- | --- | --- |
| `marker_format` | `int8_tensorwise` | `convrot_w4a4` |
| 权重存储 | int8 `[out, in]` | int8 `[out, in//2]`（打包） |
| `weight_scale` | fp32 `[out, 1]` | fp32 `[out]`（**1 维**） |
| `quant_max` | 127 | 7 |
| convrot | 可选，`in % gs == 0` 时才开，否则 marker 省略 convrot 字段 | 内建，`in % 256 != 0` 直接报错 |
| 额外约束 | K 需 8 对齐（`_int_mm`） | K % 2 == 0（打包）、K % 64 == 0（quant group）、K % 256 == 0 |
| 额外 marker 字段 | — | `convrot_groupsize`；`linear_dtype`（默认 int4，≠ 时才写） |

**W4A4 的 scale 形状不确定性**（诚实标注）：comfy-kitchen 的 eager 实现是 per-row `[out]`，但它的 `quant_group_size=64` 参数暗示 CUDA 内核可能用 per-group-64 缩放（那部分是编译好的 `.so`，读不到），而且目前**没有任何公开的 `convrot_w4a4` checkpoint 可以对照**。因此 `ConvRotW4A4Scheme` 在加载时**按 `weight_scale.numel()` 嗅探**：`== out` → per-row；`== out * in/64` → per-group-64；其它 → 报错。这样等真出现 checkpoint 时不用改结构。

扩展位：未来加 `AWQW4A16Scheme` / `SVDQuantW4A4Scheme` 只需实现同一组方法 + 一条 `register_quant_method`；`float8/mxfp8/nvfp4` 也复用同一套 marker 机制（它们只是 `convrot=False` 且 scale 语义不同），但**本次不实现**，避免摊薄验证深度。

### 7.5 `ConvRotLinear(nn.Linear)` —— 逐项实现细节

```python
class ConvRotLinear(nn.Linear):
    dtype_guarded_tensor_names = ("weight", "weight_scale")

    def __init__(self, in_features, out_features, bias, *, scheme, convrot,
                 convrot_groupsize, compute_dtype, full_precision_mm=False):
        with torch.device("meta"):
            super().__init__(in_features, out_features, bias=bias, dtype=compute_dtype)
        del self.weight                                    # 丢掉 nn.Linear 建的 fp Parameter
        self.register_buffer("weight", torch.empty(*scheme.storage_shape(out_features, in_features),
                                                  dtype=scheme.storage_dtype, device="meta"))
        self.register_buffer("weight_scale", torch.empty(*scheme.scale_shape(out_features, in_features),
                                                        dtype=torch.float32, device="meta"))
        ...
```

- **为什么能安全继承 `nn.Linear`**：`nn.Linear(..., device="meta")` 是官方支持的（`reset_parameters` 在 meta 上是空操作）；`del self.weight` 走 `nn.Module.__delattr__`，把条目从 `_parameters` 摘掉，之后同名 `register_buffer` 合法。`in_features` / `out_features` / `extra_repr` 全部保留。
- **weight 用 buffer 而不是 Parameter**：避免 int8 张量出现在 `model.parameters()`（优化器/DeepSpeed ZeRO3 会困惑）。`bias` 保持 `nn.Linear` 建的 Parameter，与 bnb 行为一致。
- **`_apply` dtype 守卫（必需）**：`load_model` 里有 `model.to(dtype=torch_dtype/offload_dtype)`，不守卫会把 int8 权重和 fp32 scale 直接改型改废。照抄 `Fp8Linear._apply` 的写法：包一层 `fn`，凡是 `dtype_guarded_tensor_names` 里的张量若被改了 dtype，就退化成"只换 device"。这是量化后端契约 (b) 的要求。
- **`forward` 三档**（按序判定，判定结果按 `(device.type, dtype)` 缓存以免每步重算）：

  1. `full_precision_matrix_mult`（marker 置位）或非 CUDA 或 `SM < 7.5` 或 `torch.is_grad_enabled() and x.requires_grad` → **反量化档**：`W_hat = rotate(dequantize_rows(...), h, gs)` 后 `F.linear`。可微、无 shape 约束。
  2. 其余 → **低比特 GEMM 档**：
     ```
     x_rot = rotate_last_dim(x, h_x, gs)                  # h_x 用 x.dtype 构建
     x8, x_scale = scheme.quantize_activation(x_rot)       # int8 容器，range 由 scheme 决定
     x8 = pad_rows(x8, to=max(M,32) 向上取 32)              # ★ M>16 约束
     acc = torch._int_mm(x8, scheme.to_int8_operand(self.weight).T)   # ★ 必须是 .T 视图
     out = (acc * (x_scale * self.weight_scale.reshape(1, -1))).to(x.dtype)
     out = out[:M] + bias
     ```
     `to_int8_operand` 对 int8 scheme 是恒等（零拷贝）；对 w4a4 是 unpack（一次性结果可缓存，避免每步解包）。**激活侧不做打包**：int4 值装在 int8 容器里直接喂 `_int_mm`，数学等价且省两次位运算 —— 这正是 comfy 的 `linear_dtype="int8"` 路径。
  3. 兜底：`_int_mm` 抛异常（罕见 shape）→ 落到第 1 档并 warn once。
- **`extra_repr`**：追加 `format=... convrot=... groupsize=...`，方便 `print(model)` 排查。
- **LoRA**：推理期 LoRA 由 `AutoWrappedQuantizedModule` 的 `LoRAHotLoadMixin` 处理（在 x 的 dtype 下算 `x @ A.T @ B.T`，与基权重 dtype 无关），不需要 Linear 自己管。

**继承 `nn.Linear` 带来的三个连带效应（已核查）**：

| 位置 | 效应 |
| --- | --- |
| `training_module.py:231` `auto_detect_lora_target_modules` | `isinstance(module, nn.Linear)` ✅ 能识别量化层（正面）。但 `linear_detector=lambda x: min(x.weight.shape) >= 512` 对 **w4a4 看到的是打包后的 `[out, in//2]`**，极端窄层可能误判 —— 需在文档里注明，或用 `in_features/out_features` 判定。 |
| peft `inject_adapter_in_model` | peft 0.20.0 的 `tuners_utils.py:1749` 有 `if base_layer_dtype.is_floating_point or ...` 守卫，int8 基权重只做 device 迁移不做 dtype 转换 ⇒ 注入**应该**可行，但**必须在训练阶段实测**，不列入第一阶段承诺。 |
| `QuantizeConfig._should_quantize` | `isinstance(module, nn.Linear)` 为真 ⇒ 对同一模型重复调用 `quantize_model()` 会二次量化（bnb 的 `Linear4bit` 早就有同样暴露）。建议加一行守卫，见 §8.4-③。 |
| `enable_vram_management_recursively` | quantize 分支在 module_map 循环**之前**（layers.py:563），所以不会被 `AutoWrappedLinear` 抢走 ✅ 无需改动。 |

### 7.6 Backend 方法逐项

```python
@register_quant_backend("convrot")
class ConvRotQuantBackend(QuantBackend):
```

| 方法 | 实现要点 |
| --- | --- |
| `capabilities()` | `is_serializable=True`（有 flatten/unflatten）、`is_differentiable=True`（反量化档可微，`check_differentiable` 能过）、`is_compileable=True`、`requires_calibration=False` |
| `validate_environment()` | **不能硬要求 CUDA**：`load_prequantized` + CPU offload 是合法用法。只在真正走 GEMM 档时检查 `torch._int_mm` 与 SM ≥ 7.5，检查失败就降级并 warn once |
| `create_quantized_linear_shell(linear, compute_dtype)` | 在 meta 上建 `ConvRotLinear`，scheme / convrot / groupsize 来自 backend config。**shell 阶段还看不到 marker**，所以配置来自注册条目；真值由 `unflatten_state_dict` 校验后回填（不一致直接报错） |
| `quantized_linear_classes()` | `(ConvRotLinear,)` |
| `unflatten_state_dict(sd, metadata)` | 逐层 pop `<layer>.comfy_quant` → 解析（扁平/嵌套双形态）→ 校验 `format` 属于本 scheme、`convrot_groupsize` 与配置一致、`weight`/`weight_scale` 都在 → 把 `full_precision_matrix_mult` 记到旁路表供 Linear 读取 → 返回不含 marker 的 sd。**这是唯一能看到原始 state_dict 的钩子，marker 的全部责任都在这里。** |
| `flatten_state_dict(sd)` | 逆操作：为每个量化层重建 marker 字节（严格按 §3.2 的写入规则），产物能被 stock ComfyUI 直接读 |
| `dequantize_to_linear(module, compute_dtype, ...)` | fp32 反量化 + 逆旋转 → 普通 `nn.Linear`（支撑 `mode="dequant_once"`，也是精度对照基线） |
| `create_quantized_linear(linear, ...)` | 在线量化（absmax 配方），用于我们自己产 checkpoint。**注释里写明**：与 §5.2 的第三方 artifact 不 bit 相同（对方多做 clipping 搜索）；`in % gs != 0` 的层按 scheme 规则退化或报错 |
| `checkpoint_key_patterns(module)` | `("weight", "weight_scale", "comfy_quant", "bias")` —— §8 的新接口 |

### 7.7 注册条目

```python
register_quant_method("comfy_int8_convrot", "convrot",
    _scheme_config(Int8TensorWiseScheme, convrot=True,  groupsize=256),
    label="8bit, int8 W8A8 + group-256 Hadamard rotation (ComfyUI int8_tensorwise)")

register_quant_method("comfy_int8_tensorwise", "convrot",
    _scheme_config(Int8TensorWiseScheme, convrot=False),
    label="8bit, int8 W8A8, no rotation (ComfyUI int8_tensorwise)")

register_quant_method("comfy_convrot_w4a4", "convrot",
    _scheme_config(ConvRotW4A4Scheme, convrot=True, groupsize=256),
    label="4bit, packed int4 W4A4 + group-256 Hadamard rotation (ComfyUI convrot_w4a4; large accuracy loss)")

register_quant_method("comfy_convrot_w4a4_int8mm", "convrot",
    _scheme_config(ConvRotW4A4Scheme, convrot=True, groupsize=256, linear_dtype="int8"),
    label="4bit weights, int8 MMA path (ComfyUI convrot_w4a4 linear_dtype=int8)")
```

`backend_config_kwargs` 暴露：`convrot`、`convrot_groupsize`、`linear_dtype`、`full_precision_matrix_mult`、`dequantize_math_dtype`（fp32/bf16，见 §7.2）、`kernel`（`"auto"` / `"torch"` / `"comfy_kitchen"`，见 §7.8）。`describe_quant_method("comfy_int8_convrot")` 会自动把这些默认值打出来。

### 7.8 加速路径（根据 §6.2 / §6.3 的实测）

收敛到**一个函数**里，scheme 不感知；三档按序降级：

```python
def _select_int8_gemm():
    """选一个 int8 W8A8 GEMM 实现：移植的 triton kernel > torch 。
    triton 不可用（非 CUDA / libdevice 缺失 / 编译失败）则 warn once 并永久用 torch 路径。"""
```

| 档 | 内容 | 实测 |
| --- | --- | --- |
| 默认（内置） | 移植的 triton kernel（§6.3） | **1.59×**，与上游逐元素相同 |
| 回退 | `torch._int_mm` + eager 尾巴 | 0.98×，但任何环境可跑、可微 |
| 可选 | `torch.compile(mode="max-autotune-no-cudagraphs")` 包住 eager 路径 | 1.57×，但**改变数值**（rel_l2 0.01270 vs 0.01304）且有 recompile / 动态 shape 风险 |

实测对比（同一层同一 shape，tokens=8192）：

| 路线 | 相对 bf16 | 代码量 | 与 ComfyUI 逐元素一致 |
| --- | --- | --- | --- |
| eager | 0.98× | 0 | ✅ |
| `torch.compile` 默认 | 1.30× | 1 行 | ❌（Inductor 改了精度/顺序） |
| `torch.compile` max-autotune | 1.57× | 1 行 | ❌ |
| 自写 triton rescale 尾巴 + `_int_mm` | 1.34× | ~25 行 | ≈（bf16 舍入量级） |
| 自写完整 triton int8 GEMM | 1.48× | ~60 行 | ≈ |
| **移植上游 kernel** | **1.59×** | ~215 行（抄） | **✅ max\|diff\| = 0** |

选移植的两个理由：它**同时**拿到最高性能与与 ComfyUI 的比特级一致（后者让"官方 kernel 作 oracle"的回归用例能卡 `atol=0`）；而 `torch.compile` 虽然只要 1 行，却会改变数值、且与 VRAM 管理的模块搬迁 / 动态 token 数存在 recompile 风险（MiniMax-H3 的 token 数随分辨率/帧数变）。

#### ❗ triton 的硬件/平台依赖 —— 为何它只能是"加速档"而不能是正确性路径

四层约束，逐层都会卡人：

1. **打包层**：torch 对 triton 的依赖声明是 `triton==3.6.0; platform_system == "Linux"` ⇒ **只有 Linux 的 torch 自带 triton**。Windows / macOS 的 torch **不带**。而 DiffSynth 明确支持 Windows（`core/loader/config.py` 里专门有 Windows 的下载提示分支）。
2. **编译后端层**：本机这个 wheel 实测只有 `['amd', 'nvidia']` 两个后端。**没有 Ascend NPU**（DiffSynth 的 `core/device` 支持 `npu`）、没有 Intel XPU（需单独装 intel-xpu-backend-for-triton）、没有 CPU 后端。
3. **硬件能力层（真正的约束）**：`tl.dot` 用 int8 要走 IMMA，需 **NVIDIA SM ≥ 7.5（Turing）** —— 与 `torch._int_mm` / cuBLASLt 及 ComfyUI 自己的门禁完全一致。AMD 需 matrix core：CDNA gfx9xx(MFMA) 或 RDNA3+ gfx11xx/gfx12xx(WMMA)；**RDNA1/2 (gfx10xx) 两者都没有，ComfyUI 原话是 int8 路径会"hang the GPU"**（不是报错，是挂死）。
4. **运行时层**：`import triton` 成功 ≠ 可用。实测无 GPU 时 import 能过，但 `triton.runtime.driver.active.get_current_target()` 抛 `RuntimeError: 0 active drivers`；首次 kernel 编译还可能因 ptxas / 编译环境失败，旧 triton 在 HIP 后端缺 `libdevice.rint` 会硬崩。

降级矩阵（必须全部覆盖）：

| 环境 | triton int8 路径 | 实际走哪档 |
| --- | --- | --- |
| Linux + NVIDIA SM ≥ 7.5 | ✅（sm80 已实测，`GPUTarget(backend='cuda', arch=80)`） | 移植的 triton kernel（1.59×） |
| Linux + NVIDIA SM 7.0（V100） | ⚠️ 可编但无 int8 IMMA | `torch._int_mm` 也不支持 ⇒ **反量化档** |
| Linux + AMD CDNA / RDNA3+ | ⚠️ 可能可用 | gfx 白名单 + 数值自检通过才启用，否则反量化档 |
| Linux + AMD RDNA1/2 (gfx10xx) | ❌ **会挂死 GPU** | **必须硬性排除**（按 gfx 名单）⇒ 反量化档 |
| Windows（任何卡） | ❌ torch 不带 triton | eager / 反量化档 |
| macOS / MPS | ❌ | 反量化档 |
| Ascend NPU（`IS_NPU_AVAILABLE`） | ❌ | 反量化档（这条必须留，DiffSynth 支持 npu） |
| CPU | ❌ | 反量化档 |

⇒ **两个设计约束因此确立：**

- 默认的 `torch._int_mm` 档本身也要 SM ≥ 7.5，所以**真正的兜底必须是反量化 + `F.linear`** —— 那条路任何设备（CPU / NPU / 老卡 / Windows）都能跑，也是可微的那条。
- 探测不能只看 `import`：必须是「查 target/arch → 白名单门禁 → 真编一个小 kernel → 与反量化档数值自检 → 结果缓存」，任何一步失败就 warn once 并永久降级。

---

## 8. 框架层接口（要求 ③）

### 8.1 现状与问题

```python
# core/vram/layers.py::AutoWrappedQuantizedModule
def _disk_required_keys(self):
    weight_prefix = self.name + ".weight."
    self._required_keys = [key for key in self.disk_map
                           if key == self.name + ".weight" or key.startswith(weight_prefix)
                           or key == self.name + ".bias"]
```

两个毛病：

1. 这套过滤是照 bitsandbytes 的键形（`x.weight.absmax`）**硬编码**的。convrot 的边料是 `x.weight_scale`（下划线不是点）、`x.comfy_quant`，**收不到** ⇒ disk offload 时该层只有权重没有 scale，加载即错。现有 `ideogram4_fp8`（同样是 `weight_scale`）踩在同一个坑上，只因没有 low_vram 示例而未暴露。
2. 同类问题还有两处 `self.quantize.backend.create_quantized_linear_shell(...)`（layers.py:512 / 538）：`MixedQuantizeConfig` **没有 `.backend` 属性**，所以「混合量化 + disk offload」目前直接 `AttributeError`。

### 8.2 新接口：`QuantBackend.checkpoint_key_patterns()`

```python
class QuantBackend:
    def checkpoint_key_patterns(self, module: torch.nn.Module) -> tuple[str, ...]:
        """本 backend 的一个量化 Linear，在 checkpoint 里除自身 `weight` 之外还需要哪些条目。

        返回的每一项是相对该层名的「相对键」：
          - 精确名，如 "weight_scale"；
          - 以 "." 结尾的前缀，如 "weight."，表示该前缀下的所有条目
            （bitsandbytes 把 quant state 嵌在 `<layer>.weight.absmax` 之类的键里）。

        disk offload 用它从整文件的键索引里精确取出一层；`unflatten_state_dict`
        需要的所有条目都必须列在这里。默认覆盖 bnb 式的嵌套布局。
        """
        return ("weight", "weight.", "bias")
```

- `bitsandbytes` / `torchao`：用默认值，行为与今天完全一致（向后兼容）
- `convrot`：`("weight", "weight_scale", "comfy_quant", "bias")`
- `ideogram4_fp8`：加一行 `("weight", "weight_scale", "bias")`，disk offload 顺带修好

### 8.3 config 层解析（框架只做一次字符串工作）

```python
class QuantizeConfig:
    def checkpoint_keys(self, module, layer_name: str, available_keys) -> list[str]:
        """把 backend 的相对键模式解析成 `layer_name` 下真实存在的 checkpoint 键。
        `available_keys` 可以是任何支持 `in` 与迭代的键集合（含 `DiskMap`）。
        缺少打包权重时报错——静默少读一个 scale 会让该层数值悄悄跑偏。"""

    def build_quantized_shell(self, module, compute_dtype):
        """委托给 backend 的 create_quantized_linear_shell，供 VRAM 管理重建空壳。"""

class MixedQuantizeConfig:
    # 两个方法都按 `config.is_quantized_linear(module)` 派发到拥有该层的子 config，
    # 找不到就报错（而不是猜第一个）。
```

`DiskMap` 已实现 `__iter__` / `__contains__`（`disk_map.py:83-93`），精确键走 O(1) 的 `in`，只有带 `.` 前缀的模式才需要扫一遍键表 —— 即只有 bnb 付这个代价，convrot 全是精确键。

### 8.4 需要你确认的 core 改动清单

| # | 位置 | 改动 | 必需性 |
| --- | --- | --- | --- |
| ① | `core/quant/base.py` + `core/quant/config.py` | 新增 `checkpoint_key_patterns` / `checkpoint_keys` / `build_quantized_shell`（§8.2、§8.3），共 +1 基类方法、+2×2 config 方法 | **必需**（要求 ③） |
| ② | `core/vram/layers.py` | 3 行替换：`_disk_required_keys` 改调 `self.quantize.checkpoint_keys(...)`；两处 `self.quantize.backend.create_quantized_linear_shell(...)` 改调 `self.quantize.build_quantized_shell(...)`。顺带修好 fp8 与 Mixed 的 disk offload | **必需**（要求 ③ 的落点） |
| ③ | `core/quant/config.py::_should_quantize` | 加一行 `if self.is_quantized_linear(module): return False`，防止重复调用 `quantize_model()` 把已量化层再量化一次（nn.Linear 子类的固有暴露，bnb 也有） | 建议（加固） |
| ④ | `core/quant/backends/convrot.py` | 新文件（§7）。放在 core 意味着它是共用框架代码 | 需定位（也可先落 `models/`，验证后上提） |

---

## 9. 注册与示例

```python
# diffsynth/models/minimax_h3_dit.py（或 configs 内）——201 个全名，两行生成
MINIMAX_H3_CONVROT_TARGETS = ["condition_proj"] + [
    f"blocks.{i}.{name}" for i in range(50)
    for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
]

# diffsynth/configs/model_configs.py, minimax_h3_series
{
    # ModelConfig(model_id="Gluttony10/MiniMax-H3-INT8-CONVROT",
    #             origin_file_pattern="MiniMax-H3-FL2VA-int8_convrot.safetensors")
    "model_hash": "68f08b5dd411f3798fc73f4699bb1d0e",      # FL2VA / Ref2VA 共用
    "model_name": "minimax_h3_dit",
    "model_class": "diffsynth.models.minimax_h3_dit.MiniMaxH3DiT",
    "quant_config": {"method": "comfy_int8_convrot", "load_prequantized": True,
                     "target_modules": MINIMAX_H3_CONVROT_TARGETS},
},
{
    "model_hash": "7b7cf3198d4a0522bf8892f1adcc63e1",      # qwen3-vl-32b-int8_convrot（第二阶段）
    "model_name": "minimax_h3_text_encoder",
    "model_class": "diffsynth.models.minimax_h3_text_encoder.MiniMaxH3TextEncoder",
    "state_dict_converter": "diffsynth.utils.state_dict_converters.minimax_h3_text_encoder.MiniMaxH3TextEncoderStateDictConverter",
    "quant_config": {"method": "comfy_int8_convrot", "load_prequantized": True,
                     "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]},
},
```

**为什么 DiT 必须写全名**：`_name_matches` 只支持「全名相等」或「以 `.pattern` 结尾」，**不支持前缀排除**。写 `["qkv_proj","fc1",...]` 会把 `token_refiner.blocks.*` 的 8 个 BF16 层一起套进去，加载当场挂。TE 反而可以用后缀，因为 visual tower 用的是 `qkv`/`proj`/`linear_fc1` 另一套名字。

示例脚本：`examples/minimax_h3/model_inference/MiniMax-H3-INT8-CONVROT-{FL2VA,Ref2VA}.py`，照 NF4 版改 `model_id` / `origin_file_pattern`；VAE 直接指这个仓库的单文件版（hash 已注册）。low_vram 版等 §8.4-② 落地后再加。

---

## 10. 实施与测试计划

**阶段 0（已完成）**：格式全景 + 数值验证 + 运行时选型 + 接口设计 —— 本报告。

**阶段 1：框架接口 + backend 骨架**
1. §8.4-① / ② 的接口改动（改动前再与你确认一次 diff）。
2. `convrot.py`：旋转原语 → 4bit codec → 两个 scheme → `ConvRotLinear` → backend → 4 条注册。
3. 单元冒烟（CPU/单卡秒级，全部放 `workspace/20260804_convrot_quant/`）：
   - Hadamard：对称性 `H == H.T`、正交性 `H@H ≈ I`、非 4 次幂 size 必须报错；
   - 4bit codec：随机 `[-8,7]` 往返无损、K 为奇数必须报错；
   - `ConvRotLinear` vs bf16 `F.linear`：int8 档 rel L2 ≈ 0.013、反量化档 ≈ 0.009；
   - `_int_mm` padding：M ∈ {1, 5, 16, 17} 全部不报错且数值与反量化档一致；
   - `.to(torch.bfloat16)` / `.half()` 后 `weight.dtype == int8` 且 `weight_scale.dtype == float32`（dtype 守卫）；
   - `check_differentiable(ConvRotLinear(...))` 通过（反量化档）；
   - marker 往返：`flatten → unflatten` 复原，且字节与发布文件的 marker 完全一致；缺 `format` / 陌生 `format` / groupsize 不匹配三种畸形输入都必须报错；
   - `checkpoint_keys`：bnb 键形与 convrot 键形各取一例，断言取到的键集合正确、缺 weight 时报错。
   - **官方 kernel 作 oracle（新增，有 comfy-kitchen 时自动跑，没则 skip）**：`ConvRotLinear.forward` 与 `torch.ops.comfy_kitchen.int8_linear` 逐元素相等；反量化与 `dequantize_int8_convrot_weight` 逐元素相等。已在 §5.2b 验证可行，固定成回归用例很便宜。
4. 加载验证：只加载 DiT，断言 201 层是 `ConvRotLinear`、其余仍是 `AutoWrappedLinear`，打印实测内存/显存。

**阶段 2：端到端推理**
5. 新示例脚本，先小步数（8 步）通链路，再 50 步出正式产物。
6. 与 bf16 基线同 seed 对照（bf16 DiT 在 `models/MiniMax/MiniMax-H3/FL2VA/transformer`，本地已有），记录耗时/峰值显存 + 主观画质。
7. 两张 A100 目前均空闲（0 MiB / 80 GiB，300 GB RAM / 290 GB 可用），bf16 与 INT8 两组对照可分卡并行；启动前我会再确认一次占用。

**阶段 3（可选）**：TE int8_convrot；disk offload；`torch.compile`；LoRA 训练（peft 注入实测）；`convrot_w4a4` 用自产 checkpoint 走通结构（精度按 §5.3 预期，只验证链路）；导出工具。

---

## 11. 风险

| 风险 | 评级 | 说明 / 缓解 |
| --- | --- | --- |
| 手搓 eager W8A8 没有速度收益 | 低（已有解） | 显存收益确定（36→18 GiB）；速度三档：comfy-kitchen triton 1.55× > `torch.compile` 1.30× > eager 0.84×。第一阶段只承诺显存。 |
| `_int_mm` 布局/尺寸约束 | 中 | 必须 `q.T` 视图 + M padding，已进单测清单。 |
| `convrot_w4a4` 无公开 checkpoint 可对照 | 中 | scale 形状按 numel 嗅探（§7.4）；精度按论文本意必须配混合精度（§5.3），第一阶段不做。 |
| peft LoRA 注入到 `nn.Linear` 子类（int8 buffer） | 中 | peft 0.20.0 有浮点守卫，理论可行，阶段 3 实测；`min(weight.shape)` 检测器对 w4a4 打包 shape 需注意。 |
| 重复 `quantize_model()` 二次量化 | 低 | §8.4-③ 一行守卫。 |
| 精度略差于 W8A16 | 低 | 0.0136 vs 0.0091；W8A8 才是 ComfyUI 真实语义，阶段 2 同 seed 主观确认。 |
| 无法 bit 复现第三方 artifact | 低 | 已用**官方 kernel 交叉验证**：官方量化器也只到 48.6362%，与我手搓一位不差 ⇒ 是对方多做了裁剪搜索，不是我们的问题（§5.2）。 |
| comfy-kitchen 依赖 | 低 | **不做硬依赖**，仅可选加速 + 一次性数值自检（§7.8）。已实测：cp310 wheel 在 torch 2.10.0+cu128 下 import 正常、triton 路径可用且快；**cuda 路径报 available 但实际不可用**（需 cu13）—— 这就是为何必须做自检而不能信 `list_backends()`。 |
| 论文官方代码不可得 | 低 | 不影响实现：算法细节论文已讲清，且有 Comfy Org 官方 kernel 可作 oracle。将来若开源，可用作第二个参考实现。 |

---

## 12. 附录：复现命令与参考

```bash
PY=/root/miniconda3/envs/debug/bin/python      # torch 2.10.0+cu128
cd /mnt/nas1/zhanghong/project26/main_project/DiffSynth-Studio

CUDA_VISIBLE_DEVICES=0 $PY workspace/20260804_convrot_quant/probe_convrot_format.py   # marker 一致性 + 反量化对齐 bf16
CUDA_VISIBLE_DEVICES=0 $PY workspace/20260804_convrot_quant/probe_quant_recipe.py     # 能否 bit 复现生产方配方
CUDA_VISIBLE_DEVICES=0 $PY workspace/20260804_convrot_quant/probe_scale_origin.py     # 定位 clipping 搜索
CUDA_VISIBLE_DEVICES=0 $PY workspace/20260804_convrot_quant/probe_runtime_options.py  # 三种运行时 + _int_mm 布局对比
CUDA_VISIBLE_DEVICES=0 $PY workspace/20260804_convrot_quant/probe_w4a4_codec.py       # W4A4 codec 与精度
CUDA_VISIBLE_DEVICES=0 $PY workspace/20260804_convrot_quant/probe_hadamard_variants.py # regular vs Sylvester vs random（能否复用现成库）

# 官方 comfy-kitchen kernel 对比（不安装到环境，只用 PYTHONPATH 挂载解包后的 wheel）
pip download comfy-kitchen --no-deps -d /tmp/ck310 && (mkdir -p /tmp/ck310_x && cd /tmp/ck310_x && unzip -o -q /tmp/ck310/*.whl)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/ck310_x $PY workspace/20260804_convrot_quant/probe_comfy_kitchen.py                  # cuda 路径（本机失败）
CUDA_VISIBLE_DEVICES=0 DISABLE_CK_CUDA=1 PYTHONPATH=/tmp/ck310_x $PY workspace/20260804_convrot_quant/probe_comfy_kitchen.py  # triton 路径（可用，1.55×）
```

权威参考：

- **论文（算法权威定义）**：ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers，`https://arxiv.org/abs/2512.03673`（2025-12-03）。**未找到作者公开的官方代码仓库**（abs 页无 code 链接；OpenReview `SCC11m676G` 被人机验证拦截）。
- **事实标准实现（Comfy Org 官方，Apache-2.0）**：`comfy-kitchen`（PyPI，本次用 0.2.26）的 `tensor/int8.py`、`tensor/int8_utils.py`、`tensor/convrot_w4a4.py`、`backends/eager/{quantization,convrot_w4a4,svdquant}.py`
- `QUANT_ALGOS` 与 layout 注册：`https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/quant_ops.py`
- marker 读写实现：`https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ops.py`（`_load_quantized_weight` / `_quantized_weight_state_dict`）
- 格式契约与量化配方：`https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/int8_tensorwise.md`
- 生产方说明：`models/Gluttony10/MiniMax-H3-INT8-CONVROT/README.md`
- 前驱工作：QuaRot（论文明言 kernel 基于它）、SVDQuant（主要对标，需专用推理引擎）、QuIP#
