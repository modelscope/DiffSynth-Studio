# ConvRot 后端依赖报告

- 日期：2026-08-04
- 范围：`diffsynth/core/quant/backends/convrot.py`（拟新增）所支持的全部 config，以及每个 config 在**量化 / 反量化 / 前向计算**三类操作上的软硬件依赖
- 配套：《ConvRot 通用量化后端调研报告》（`CONVROT_INT8_INTEGRATION_REPORT.md`）给格式与设计，本报告只讲依赖
- 数据来源：`workspace/20260804_convrot_quant/` 下的探测脚本，实测机器为 A100-80GB / torch 2.10.0+cu128 / triton 3.6.0 / driver 570.133.20
- 所有计时均在 **GPU 空闲时**采集（早前一组在卡被占用时采到的数字已作废并重测）

---

## 0. 摘要

| 结论 | |
| --- | --- |
| 第三方依赖 | **零**。不需要 bitsandbytes / torchao / comfy-kitchen / fast-hadamard-transform / scipy |
| 必需依赖 | 只有 `torch` + `safetensors`（DiffSynth 本来就依赖） |
| 可选依赖 | `triton`（Linux 上 torch 自带，**不是新增依赖**）；装了才有 1.59× 加速档 |
| 最低可用门槛 | **CPU 也能跑**（反量化档），因此 Windows / macOS / Ascend NPU / 老卡全部可用 |
| 加速门槛 | Linux + NVIDIA SM ≥ 7.5（或 AMD matrix-core），三层门禁 + 自检才启用 |
| 最大的坑 | **`int8_tensorwise` 与 `convrot_w4a4` 的依赖档位完全不同**：前者能在纯 Python 上跑满，后者的真 int4 GEMM 在纯 Python 里**根本不可达**（详见 §7） |

---

## 1. 支持的 config 清单

config 由三个正交维度构成，共同决定依赖：

**维度 1：`method`（4 个注册条目）**

| method | 上游 `format` | 权重存储 | scale | GEMM 语义 |
| --- | --- | --- | --- | --- |
| `comfy_int8_convrot` | `int8_tensorwise` + convrot | int8 `[out, in]` | fp32 `[out, 1]` | W8A8 + 在线激活旋转 |
| `comfy_int8_tensorwise` | `int8_tensorwise`（无旋转） | int8 `[out, in]` | fp32 `[out, 1]` | W8A8，无旋转 |
| `comfy_convrot_w4a4` | `convrot_w4a4` | int8 `[out, in/2]`（打包 nibble） | fp32 `[out]` | W4A4 + 在线激活旋转 |
| `comfy_convrot_w4a4_int8mm` | `convrot_w4a4`, `linear_dtype="int8"` | 同上 | 同上 | 4bit 权重走 int8 MMA |

**维度 2：`mode`**

| mode | 含义 | 对依赖的影响 |
| --- | --- | --- |
| `dynamic`（默认） | 保持低比特权重，每次 forward 走量化路径 | 需要 §4 的前向依赖 |
| `dequant_once` | 加载后一次性还原成普通 bf16 `nn.Linear` | **前向依赖归零**，退化成普通模型；只在加载时用一次反量化 |

**维度 3：`load_prequantized`**

| | 含义 | 对依赖的影响 |
| --- | --- | --- |
| `True` | 读已有 convrot checkpoint | **不需要量化算子**，只需加载 + 前向 |
| `False` | 在线量化 fp 权重 | 需要 §2 的量化依赖 |

⇒ 实际组合中最常用的三种：

- **A. 推理**：`method=comfy_int8_convrot, mode=dynamic, load_prequantized=True` ← MiniMax-H3 的主用例
- **B. 精度对照基线**：`mode=dequant_once, load_prequantized=True`
- **C. 自产 checkpoint**：`mode=dynamic, load_prequantized=False` + `flatten_state_dict` 存盘

---

## 2. 操作一：离线量化（fp 权重 → 低比特）

触发路径：`QuantizeConfig.quantize_model()` → `backend.create_quantized_linear()`；以及导出工具。

### 算子分解

| 步骤 | 用到什么 | 依赖等级 |
| --- | --- | --- |
| 构造 regular Hadamard | `torch.kron` + 标量除法 | 任意设备 |
| 权重旋转 `W.view(out, g, gs) @ H` | `torch.matmul`（fp32 或 bf16） | 任意设备 |
| per-row absmax | `abs().amax(-1)` | 任意设备 |
| 除 + round + clamp + cast | 逐元素算子 | 任意设备 |
| int4 打包（仅 w4a4） | `&` `\|` `<<` 位运算（int32 中转） | 任意设备 |

### 依赖表

| config | 软件依赖 | 硬件依赖 | 备注 |
| --- | --- | --- | --- |
| `comfy_int8_convrot` | torch 基础算子 | **无特殊要求，CPU 可完成** | 大权重在 CUDA 上快得多，但非必需 |
| `comfy_int8_tensorwise` | 同上（不建 Hadamard） | 同上 | |
| `comfy_convrot_w4a4` / `_int8mm` | 同上 + 位运算 | 同上 | 打包要求 `in % 2 == 0` |

**要点**

- 量化是这三类操作里**依赖最轻**的一环：没有 tensor core、没有 triton、没有第三方库。
- 唯一的"软"要求是**内存**：旋转与 absmax 需要一份 fp32 的权重副本。MiniMax-H3 单层最大 `[28672, 5376]` = fp32 588 MiB，逐层做即可，不需要整模型驻留。
- 数值约定必须与上游一致（`quant_max` int8=127 / int4=**7 不是 8**，fp32 与 bf16 的除法位置），否则产物 ComfyUI 读不了。已验证我们的配方与官方 kernel **逐位一致**（同层码字一致率 0.486362 完全相同，双方都无法复现那个第三方 artifact，因为对方多做了裁剪搜索）。

---

## 3. 操作二：加载与反量化

### 3.1 预量化加载（`load_prequantized=True`）

| 步骤 | 用到什么 | 依赖 |
| --- | --- | --- |
| 读 safetensors | `safetensors.safe_open` | 无 |
| 解析 `comfy_quant` marker | `json.loads(bytes)` | 无 |
| 建 meta 壳 | `torch.nn.Linear(..., device="meta")` | 无 |
| `load_state_dict(assign=True)` | torch | 无 |

**这一步零计算、零 GPU。** 任何平台（含 Windows / NPU / CPU-only）都能完成加载 —— 加载成功与能否加速是两件独立的事，这点在文档里要写清楚，避免用户以为"加载不报错就有加速"。

### 3.2 反量化（`dequantize_to_linear` / `mode="dequant_once"` / 前向的兜底档）

| 步骤 | 用到什么 | 依赖 |
| --- | --- | --- |
| int4 解包（仅 w4a4） | 位运算 | 任意设备 |
| `q.float() * scale` | 逐元素 | 任意设备 |
| 逆旋转（`@ H`，H 对称所以同一函数） | `torch.matmul` fp32 | 任意设备 |

| config | 软件依赖 | 硬件依赖 | 实测成本 |
| --- | --- | --- | --- |
| 全部 4 个 | torch 基础算子 | **无，CPU 可用** | 单层权重反量化：bf16 数学 **1.81 ms** / fp32 数学 **6.26 ms**（A100，`[28672,5376]`） |

**要点**

- 逆旋转的数学 dtype 是个明确的取舍：**fp32 是正确选择**（bf16 逆旋转额外引入 0.32% 权重误差）。只在"每次 forward 现场反量化"的热路径上才考虑 bf16 换速度，且必须是显式开关。
- `mode="dequant_once"` 用完这一步之后，模型就是普通 bf16 模型，**后续前向的所有依赖归零** —— 这是最保守的部署方式，代价是没有显存收益。

---

## 4. 操作三：前向计算（依赖的分水岭）

前向有三档，**依赖差异全部集中在这里**。

### 4.1 三档定义

| 档 | 算法 | 可微 | 数值 |
| --- | --- | --- | --- |
| **D1 反量化档** | 逆旋转还原 fp 权重 → `F.linear` | ✅ | 与 §3.2 一致，rel_l2 ≈ 0.0091 |
| **D2 `torch._int_mm` 档** | 旋转激活 → 逐行量化 → `_int_mm` → 反缩放 | ❌ | rel_l2 = 0.01304，与 ComfyUI 逐元素相同 |
| **D3 triton 档** | 同 D2，但量化与 GEMM 各用一个融合 kernel | ❌ | rel_l2 = 0.01304，与 ComfyUI **max\|diff\| = 0** |

### 4.2 软件依赖

| 档 | 必需软件 | 版本要求 | 验证方式 |
| --- | --- | --- | --- |
| D1 | torch | 无特殊 | — |
| D2 | torch + `torch._int_mm` | `_int_mm` 私有 API，torch 2.0 起存在；新版另有公开的 `torch.int8_mm`（本机 torch 2.10 **尚无**）⇒ 代码要 `hasattr` 双探，优先公开 API | 实测 `hasattr(torch,'_int_mm')=True`、`hasattr(torch,'int8_mm')=False` |
| D3 | torch + **triton** | `triton==3.6.0; platform_system == "Linux"`（torch 的条件依赖）；`from triton.language.extra import libdevice` 需存在 | 实测本机 triton 3.6.0，编译后端仅 `['amd','nvidia']` |

### 4.3 硬件依赖

| 档 | NVIDIA | AMD | 其他 |
| --- | --- | --- | --- |
| D1 | 任意（含无 GPU） | 任意 | CPU / Ascend NPU / MPS 均可 |
| D2 | **SM ≥ 7.5**（Turing，int8 IMMA）。`_int_mm` 在 **CPU 上也能跑**（实测通过，但走标量路径，无加速意义） | ROCm 下 `_int_mm` 支持情况未验证 | NPU 不支持 |
| D3 | **SM ≥ 7.5** | 需 matrix core：CDNA gfx9xx(MFMA) 或 RDNA3+ gfx11xx/gfx12xx(WMMA)。**RDNA1/2 (gfx10xx) 会挂死 GPU**（ComfyUI 原话 "hangs the GPU"） | 全部不支持 |

### 4.4 D2 的两条形状约束（实测边界，必须在代码里处理）

| 约束 | 实测 | 处理方式 |
| --- | --- | --- |
| `M > 16` | M=16 报 `self.size(0) needs to be greater than 16`；M=17 通过 | 行 padding 到 `max(M,32)` 向上取 32 倍数，算完裁回。**`condition_proj` 吃短 prompt 会撞上** |
| B 必须 column-major | 传 `q.T` 视图 = 6.42 ms；传 `q.T.contiguous()` = **35.17 ms**（比 bf16 慢 3.5×） | 永远传视图。这也是上游 `stride_bk=weight.stride(1)` 的等价写法 |

### 4.5 各 config 的前向依赖汇总

| config | D1 | D2 | D3 | 纯 Python 能达到的最好情况 |
| --- | --- | --- | --- | --- |
| `comfy_int8_convrot` | ✅ | ✅ | ✅ | **1.59×**（D3） |
| `comfy_int8_tensorwise` | ✅ | ✅ | ✅ | 同上（少一次激活旋转，略快） |
| `comfy_convrot_w4a4` | ✅ | ❌ 无 int4 matmul | ❌ | **只有 D1**，比 bf16 慢 20%（§7） |
| `comfy_convrot_w4a4_int8mm` | ✅ | ✅（解包成 int8 后） | ✅ | int8 档速度 + int4 档精度，**每步解包仅 +9% 就能保住 4bit 显存**（§7） |

### 4.6 实测性能（int8 W8A8，`blocks.0.mlp.fc1`，tokens=8192，GPU 空闲时采集）

| 路线 | 耗时 | 相对 bf16 | 代码量 | 与 ComfyUI 逐元素一致 |
| --- | --- | --- | --- | --- |
| bf16 `F.linear` 基线 | 9.8–10.7 ms | 1.00× | — | — |
| D1 反量化档（现场反量化） | 11.41 ms | ~0.9× | 0 | — |
| D2 eager | 10.36–10.43 ms | 0.98–1.03× | 0 | ✅ |
| D2 + `torch.compile` 默认 | 8.21 ms | 1.30× | 1 行 | ❌ 数值被改（rel_l2 0.01270） |
| D2 + `torch.compile` max-autotune | 6.80 ms | 1.57× | 1 行 | ❌ 同上 |
| D3 自写 triton rescale 尾巴 | 7.97 ms | 1.34× | ~25 行 | ≈ |
| D3 自写完整 triton GEMM | 7.20 ms | 1.48× | ~60 行 | ≈ |
| **D3 移植上游 kernel** | **6.40 ms** | **1.59×** | ~215 行（抄） | **✅ max\|diff\| = 0** |
| *comfy-kitchen 上游 triton（参考）* | *6.46 ms* | *1.58×* | *依赖它* | *—* |

性能差的根因**不在 GEMM 而在激活量化**：eager 用多个 torch 算子反复读写 x，上游用一个 triton kernel 一个 program 处理一整行、一次 pass 完成。

---

## 5. 软件依赖总清单

| 组件 | 必需性 | 用在哪 | 缺失后果 |
| --- | --- | --- | --- |
| `torch`（≥ 2.0） | **必需** | 全部 | — |
| `safetensors` | **必需**（DiffSynth 已依赖） | 加载 checkpoint | — |
| `torch._int_mm` / `torch.int8_mm` | 可选 | D2 | 降级到 D1 |
| `triton`（Linux 上 torch 自带） | 可选 | D3 | 降级到 D2 |
| `triton.language.extra.libdevice` | D3 内必需 | `libdevice.rint` | **旧 triton 在 HIP 后端缺它会硬崩** ⇒ import 失败必须降级 |
| ~~bitsandbytes / torchao~~ | 不需要 | — | — |
| ~~comfy-kitchen~~ | 不需要（移植 kernel 代替） | — | — |
| ~~fast-hadamard-transform~~ | 不需要 | — | 它是 Sylvester 型，**与 ConvRot 不兼容** |
| ~~scipy~~ | 不需要 | — | `scipy.linalg.hadamard` 是 Sylvester 型，**用了会读不了 checkpoint** |

**pyproject.toml 不需要任何新增条目。**

---

## 6. 硬件依赖总清单

| 资源 | 量化 | 反量化 | D1 | D2 | D3 |
| --- | --- | --- | --- | --- | --- |
| 计算设备 | 任意 | 任意 | 任意 | CUDA（CPU 可跑但无意义） | CUDA / ROCm |
| int8 tensor core | — | — | — | **SM ≥ 7.5** | **SM ≥ 7.5** 或 AMD matrix core |
| 显存（单层峰值，`[28672,5376]`） | fp32 副本 588 MiB | 同左 | 每步 transient bf16 294 MiB | int8 147 MiB 常驻 | 同 D2 |
| 主机内存 | 逐层流式即可 | 同左 | — | 43.8 GiB DiT 放 CPU（本机 300 GB 可用） | 同 |

MiniMax-H3 DiT 的实际账：201 个量化层 bf16 约 36 GiB → int8 约 18 GiB（未量化的 adaLN 25.7 GiB 不变，整文件 43.8 GiB）。

---

## 7. `convrot_w4a4` 的特殊问题：真 int4 GEMM 不可达

这是本报告最重要的一条依赖发现。

### 事实链

1. **torch 没有 int4 矩阵乘**。没有 `int4_mm`，`tl.dot` 也不支持 int4 类型。
2. **上游自己的 triton 后端也没有 W4A4**：实测 `ck.list_backends()` 的 capability 列表里，`convrot_w4a4_linear` 只出现在 **cuda** 与 **eager** 后端，**triton 后端没有这一项**。
3. **上游的 eager 实现根本不是低比特 GEMM**：`int4_linear()` 把权重与激活都解包成 `out_dtype`（浮点）再做普通 `@` —— 也就是说 eager 档的 W4A4 比 bf16 更慢、更费内存。
4. 于是真正的 int4 MMA 只存在于它的 **cuda 后端**，而那是 135 MB 的预编译 `.so`，**需要 cu13**，本机（torch cu128 / driver 570.133.20）实测直接报 `CUDA driver version is insufficient`。

⇒ **纯 Python 环境下，`comfy_convrot_w4a4` 拿不到任何 4bit 加速。** 能做的只有两条：

| 路线 | 速度 | 显存 | 说明 |
| --- | --- | --- | --- |
| 解包成 int8 → 走 D2/D3 | int8 档速度（每步解包 +9%） | **可保住 73.5 MiB** | 就是 `linear_dtype="int8"`，上游也有这条（`prepare_int4_weight_for_int8_linear`） |
| 反量化 → `F.linear`（D1） | 11.60 ms（比 bf16 慢 20%） | 常驻 int4 + transient bf16 | 唯一"能跑"的保底 |

### 显存账与解包成本（已在空闲卡上实测）

| 形态 | `[28672, 5376]` 一层 |
| --- | --- |
| bf16 | 294.0 MiB |
| int8 | 147.0 MiB |
| **int4 打包**（磁盘 / 常驻） | **73.5 MiB** |
| **解包后**（int8 MMA 实际吃的） | **147.0 MiB** |

解包成本（这里有个 **7× 的实现陷阱**）：

| 实现 | 耗时 |
| --- | --- |
| 上游同款写法（int32 中转 + `torch.stack` + reshape） | **6.75 ms** |
| **全程 int8 位移位**（`(p << 4) >> 4` 取低 nibble、`p >> 4` 取高 nibble，靠 int8 的算术右移自动符号扩展） | **0.96 ms** |
| 同上 + 复用输出 buffer | 0.96 ms（无额外收益，瓶颈在 `[..., 0::2]` 的跳写而非分配） |

⇒ **实现时必须用 int8 位移位版**，不能照抄上游的 int32+stack 写法（它把内存流量放大 4 倍）。已断言两者逐位相同。

### W4A4 前向实测（tokens=8192，eager 尾巴，GPU 空闲）

| 路线 | 耗时 | 运行时显存 |
| --- | --- | --- |
| bf16 `F.linear` 基线 | 9.71 ms | 294 MiB |
| int4 → int8 MMA，**解包一次并缓存** | 10.17 ms | 147 MiB（**4bit 收益在运行时丢失**） |
| int4 → int8 MMA，**每步解包** | **11.08 ms** | **73.5 MiB（保住 4bit）** |
| 反量化 → bf16 `F.linear`（D1） | 11.60 ms | 73.5 MiB + transient bf16 |

**这组数字改变了默认选择**：每步解包比缓存只贵 **0.91 ms（+9%）**，却把常驻显存从 147 MiB 降到 73.5 MiB。⇒ `comfy_convrot_w4a4_int8mm` 应该**默认每步解包**（才能兑现 4bit 的意义），把"解包一次并缓存"做成显式的速度优先开关。

注：上表是 eager 档。若 GEMM 换成移植的 triton kernel（~6.4 ms），解包的 0.96 ms 占比会从 9% 升到 ~15%，但结论不变。

### 结论

`convrot_w4a4` 应当**只实现存储与正确性，不承诺性能**；文档明确写"4bit 需要混合精度（论文自己的做法：20% 敏感层 INT8 + 其余 INT4），且在纯 Python 环境下无 4bit 加速"。W4A4 单独用的精度也不可接受（权重 rel_l2 0.160、端到端 0.228，比 int8 差 17×）。

---

## 8. 平台情景分析

| 平台 | 加载 | 量化 | 反量化 | D1 | D2 | D3 | 实际得到什么 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Linux + NVIDIA SM ≥ 7.5**（Turing/Ampere/Ada/Hopper/Blackwell） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **完整能力**：显存减半 + 1.59× 加速 |
| Linux + NVIDIA SM 7.0（V100） | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | 显存减半，速度 ≤ bf16 |
| Linux + NVIDIA SM 6.x（P100 等） | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | 同上 |
| Linux + AMD CDNA（MI200/MI300, gfx9xx） | ✅ | ✅ | ✅ | ✅ | ⚠️ 未验证 | ⚠️ 白名单+自检 | 需实测；保守起见默认 D1 |
| Linux + AMD RDNA3/4（gfx11xx/12xx） | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ 白名单+自检 | 同上 |
| **Linux + AMD RDNA1/2（gfx10xx）** | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ **会挂死 GPU** | **必须按 gfx 名单硬性排除 D3** |
| **Windows + 任意 GPU** | ✅ | ✅ | ✅ | ✅ | ✅（若 SM≥7.5） | ❌ **torch 不带 triton** | 显存减半，速度 ~1.0×（可试 `torch.compile`） |
| macOS / MPS | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | 仅显存收益 |
| **Ascend NPU**（DiffSynth 支持 `npu`） | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | 仅显存收益；`IS_NPU_AVAILABLE` 时直接锁 D1 |
| CPU only | ✅ | ✅ | ✅ | ✅ | ⚠️ 能跑但无意义 | ❌ | 能加载、能验证正确性 |

**跨平台的两条硬结论**

1. **加载与正确性是全平台的；加速是少数平台的。** 文档与日志必须把这两件事分开表述，否则 Windows / NPU 用户会以为坏了。
2. **默认档不能是 D2。** `_int_mm` 自己就要 SM ≥ 7.5，所以真正的兜底只能是 **D1（反量化 + `F.linear`）** —— 它同时也是唯一可微的档，LoRA 训练必须走它。

---

## 9. 探测与降级协议

`import` 成功 ≠ 可用（实测：无 GPU 时 `import triton` 能过，但 `triton.runtime.driver.active.get_current_target()` 抛 `RuntimeError: 0 active drivers`）。因此按序做：

```
1. 设备类型：npu / mps / cpu            -> 锁 D1，不再探测
2. grad 模式（requires_grad 活跃）      -> 本次 forward 用 D1（可微）
3. marker 里 full_precision_matrix_mult -> 锁 D1
4. NVIDIA: get_device_capability() >= (7,5) ?  否 -> 锁 D1
   AMD:    gcnArchName 在白名单内 ?           否 -> 锁 D1（gfx10xx 必须排除）
5. D2 可用性：hasattr(torch,'int8_mm') or hasattr(torch,'_int_mm')  否 -> 锁 D1
6. D3 可用性：platform=Linux and import triton ok and libdevice ok
              and get_current_target() 不抛异常
              and 真编一个小 kernel 并与 D1 数值自检通过         否 -> 用 D2
7. 结果按 (device, dtype, arch) 缓存；任何一步失败 warn once，永久降级，不重试
```

`kernel` 配置项暴露 `"auto"`（默认，按上面决策）/ `"torch"`（强制 D2）/ `"triton"`（强制 D3）/ `"dequant"`（强制 D1），便于排障与做数值对照。

---

## 10. 待办与待重测

| 项 | 状态 |
| --- | --- |
| W4A4 前向三条路线的端到端耗时、int4 解包耗时 | ✅ 已在空闲卡上重测完成（§7），并发现解包实现的 7× 优化空间 |
| W4A4 接 triton kernel 后的端到端耗时 | 未测（预期解包占比升到 ~15%） |
| ROCm 上 `torch._int_mm` 与 triton int8 路径 | 无硬件，未验证 |
| AMD gfx 白名单的具体名单 | 需按 ComfyUI 的 `_rocm_kitchen_arch_supported()` 抄：允许 `gfx11*`/`gfx12*` 与 `gfx908/90a/940/941/942/950` |
| Windows 上 `torch.compile` 能否补上加速 | 未验证（Inductor 在 Windows 上需 MSVC 工具链） |
| peft LoRA 注入到 `ConvRotLinear`（int8 buffer） | 未验证，属训练阶段 |

---

## 11. 附录：复现命令

```bash
PY=/root/miniconda3/envs/debug/bin/python
cd /mnt/nas1/zhanghong/project26/main_project/DiffSynth-Studio

# 依赖事实（API 存在性、CPU _int_mm、int4 内存账）
CUDA_VISIBLE_DEVICES=0 $PY workspace/20260804_convrot_quant/probe_dependency_facts.py

# 三档前向的性能与数值
CUDA_VISIBLE_DEVICES=0 $PY workspace/20260804_convrot_quant/probe_runtime_options.py
CUDA_VISIBLE_DEVICES=0 $PY workspace/20260804_convrot_quant/probe_python_kernels.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/ck310_x $PY workspace/20260804_convrot_quant/probe_ported_kernels.py

# 为何不能复用现成库的 Hadamard
CUDA_VISIBLE_DEVICES=0 $PY workspace/20260804_convrot_quant/probe_hadamard_variants.py

# triton 的平台/硬件能力
$PY -c "import triton; print(triton.runtime.driver.active.get_current_target())"
$PY -c "import importlib.metadata as m; print([r for r in m.requires('torch') if 'triton' in r])"
```
