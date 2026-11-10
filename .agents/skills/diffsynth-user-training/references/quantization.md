# 量化底模与 LoRA

仅在量化、预量化 checkpoint 或相关 OOM 时读取。依据当前 `diffsynth/diffusion/training_module.py`、`core/quant/{config,base}.py`、各 backend、`models/model_loader.py`、`core/loader/model.py` 和 `docs/zh/Pipeline_Usage/Quantization.md`，按实际版本核对行为。

## 训练语义与加载链

典型 QLoRA：浮点权重加载 → 目标 `nn.Linear` 在线量化为 backend Linear → 冻结底模 → 注入浮点 LoRA → 梯度穿过冻结层、更新 A/B。这是 LoRA 的资源配置，不是 packed 基座全量训练。即使 backend 可微，也不表示量化权重支持全量更新。全量训练可量化其他确实冻结的组件，但保留目标组件的可训练浮点权重。

`--quant_options → parse_model_configs()/parse_quant_options() → ModelConfig(quantize=...) → Pipeline.download_and_load_models() → ModelPool.resolve_quant_config() → core.loader.load_model()`。在线量化的 `N nn.Linear layers quantized (method: ...)` 是转换执行信号；N=0 或未命中必须调查。预量化加载走 `prepare_for_prequantized_load()` 等路径，不能要求它同样打印在线量化行。

`mode` 控制最终模块表示，`load_prequantized` 控制权重来源，两者是独立维度：`mode="dynamic"` 保留 backend 原生量化 Linear，既可来自在线转换（`load_prequantized=False`），也可来自预量化 checkpoint（True）；`mode="dequant_once"` 在转换/加载后还原普通浮点 Linear，失去持续压缩存储收益，但不能恢复量化前的原始数值。这也不同于每次 forward 的动态激活量化。backend 可按实现反量化或执行低精度算子，不能承诺所有方法都加速；在线转换需要浮点暂存，最终模型能放下也可能加载时 OOM。

## CLI 语法与精确匹配

### 分号格式

```text
--quant_options "<模型条目>:<method>[/<排除层1>,<排除层2>];<另一个模型条目>:<method>"
```

用**最后一个 `:`** 分离模型条目和 spec，spec 的第一个 `/` 分离 method 与排除列表；空条目跳过，空 spec（`模型条目:`）跳过，不要加入未经处理的空格。分号格式默认 `mode="dynamic"`、`load_prequantized=False`，只填写 method 和可选 exclude。

模型条目与 `--model_id_with_origin_paths` 的 `repo:pattern` 或 `--model_paths` 的组件条目对应；不是 `dit` 等 Pipeline 属性名。repo、路径与通配写法要一致。**本地一个组件的多分片必须整体匹配，且顺序一致**；当前 `normalize_quant_key()` 将 JSON 列表规范为 tuple，JSON 内空格无需逐字一致：

```bash
--model_paths '[["/models/dit-1.safetensors","/models/dit-2.safetensors"],"/models/vae.safetensors"]' \
--quant_options '["/models/dit-1.safetensors", "/models/dit-2.safetensors"]:bitsandbytes_nf4'
```

这是参数片段，需接在目标训练入口及其他必需参数后；它不声称该示意路径是已注册模型。只写一个 shard、颠倒顺序或换成本地 glob 字符串不会自动匹配列表。`No quant option matches ...` 表示对应组件没有显式量化选项（其他冻结组件不需要量化时可以正常出现），目标组件出现该提示时须修正。上述规范化仅适用于量化；FP8/Disk 的本地分片限制见 [memory.md](memory.md)。

### JSON 格式与混合配置

JSON 格式可以表达更多配置字段。例如 Qwen-Image-2512 的 transformer：

```bash
--quant_options '{"Qwen/Qwen-Image-2512:transformer/diffusion_pytorch_model*.safetensors": {"method": "bitsandbytes_nf4", "backend_config_kwargs": {"blocksize": 64}, "exclude_modules": ["img_mod.1", "txt_mod.1", "norm_out.linear", "img_in", "txt_in", "proj_out", "timestep_embedder.linear_1", "timestep_embedder.linear_2"]}}'
```

单配置可使用 `method`、`mode`、`target_modules`、`exclude_modules`、`backend_config_kwargs`、`load_prequantized`；层名字段必须是 JSON 数组。`load_prequantized` 必须写布尔 `true/false`，不能写字符串、数字或 null。`mode="dequant_once"` 会还原成普通浮点 Linear 并保留量化误差，**不提供持续的权重压缩收益**。

混合配置使用 `{"模型条目":{"configs":[{...},{...}]}}`，每个子配置给出 method 与层过滤。子配置命中的层集合必须互不重叠，否则转换前报错；所有子配置 mode 相同。混合预量化时只在 wrapper 顶层设 `load_prequantized:true`，子项保持 false。它需要更多后端和匹配验收，不作为首次低显存配置的默认值。

## 方法、依赖与梯度

可用集合以当前注册和实际 backend capabilities 为准，下表用于选型而非固定白名单：

自动选择时优先目标变体已存在的官方量化训练配方。没有对应配方再筛选设备/库版本、模型形状、真实 Linear 覆盖、输入梯度、用户可接受的数值变化与端到端成本；不只根据位宽最低或标称 bpp 最小选择方法。下表只作能力参考，实际只检查候选方法及所需依赖，不要求每次训练逐一导入/测试全部后端。后端声明可微仅是筛选条件，最终要同时验证**冻结底模可传递有限输入梯度**与**LoRA 参数发生真实更新**。成功加载或一次推理不构成训练验证。

| 方法/组 | 当前训练侧能力与选用条件 |
| --- | --- |
| `bitsandbytes_nf4`、`bitsandbytes_fp4` | 声明可微的 4bit weight-only；常见在线 QLoRA 起点，仍需验证目标环境与输入梯度 |
| `torchao_int8_w8a16`、`torchao_fp8_w8a16`、`torchao_nvfp4_w4a16` | 当前 torchao 的可微 config 集合；硬件/版本还需满足具体方法 |
| `comfy_kitchen_int8_w8a8`、`comfy_kitchen_fp8_w8a8` | 当前 backend 声明可微；不要把 backend 名与 CUDA kernel 可用性等同，检查安装和实际路径 |
| `torchao_int4_w4a16`、`torchao_int8_w8a8`、`torchao_fp8_w8a8`、`torchao_int4_w4a8`、`torchao_mxfp8_w8a8`、`torchao_mxfp4_w4a4`、`torchao_nvfp4_w4a4` | 当前 capabilities 声明不可微，显式用于训练时被 parser 拒绝，即使目标是冻结组件 |
| `deep_gemm_fp8_w8a8`、`deep_gemm_fp4_w4a4` / `w4a8` | 已注册且声明可微；前者要求支持的 SM90/SM100，FP4 要求 SM100，并需实际 DeepGEMM/CUDA 工具链与形状支持 |
| `entropack_lossless_df11` / `tile_ans`、`entropack_lossy_quant` 及其 FP8/INT8 变体 | 已注册且声明可微；区分无损压缩与有损量化，按后端和 dtype 选择，验证内核、解压暂存与输入梯度 |

后两组的完整方法名查询源码/注册表，不从缩写拼 CLI。安装当前目标所需的 backend 即可；`pyproject.toml` 的 quant extra 包含多个可能有不同 CUDA 要求的库，DeepGEMM 也不由该 extra 安装，不默认全装或升级整个环境。

在目标 Python/源码中，先查询实际候选（会导入相应后端依赖）：

```python
from diffsynth.core.quant import QuantizeConfig

cfg = QuantizeConfig(method="bitsandbytes_nf4")
print(cfg.backend.capabilities())
```

需要查询后端参数时可选用 `describe_quant_method(method)`，但当前实现内部会加载全部 backend 注册模块；只需确认已选方法可用时，直接构造其 `QuantizeConfig` 并检查 capabilities 即可。仅在需要枚举/发现其他方法时查询 `QUANT_METHODS` 或调用 `backends.load_all_backends()`，不将全后端导入设为训练前提。

`parse_quant_options()` 对显式配置检查 `is_differentiable`，不可微时报错；当前取值缺省为 True，不要假定第三方未声明就已实测通过。注册预量化配置在 loader 中另行构造，不保证经过同一个训练 parser 校验；这种底模也应检查可微性。`check_differentiable()` 可在**同 dtype/设备、独立小测试层**上验证输入梯度；它会将模块转 BF16，不能随意对正式已加载模块调用而改变配置。最终仍要在真实 LoRA smoke 中验证参数更新。

## 排除规则：必须命中真正的 Linear

`QuantizeConfig._name_matches()` 是全名相等或以 `"." + pattern` 结尾，**不是前缀、子树或 glob 匹配**。模型代码里的容器名字不能代替内部 Linear 名字：

| 排除意图 | 正确的 Linear 名称示例 | 不命中的容器写法 |
| --- | --- | --- |
| Qwen 调制 Linear | `img_mod.1`、`txt_mod.1` | `img_mod`、`txt_mod` |
| Qwen 输出 norm 内部投影 | `norm_out.linear` | `norm_out` |
| timestep 内部两层 | `timestep_embedder.linear_1`、`timestep_embedder.linear_2` | `timestep_embedder` |

在量化前枚举精确变体的 `model.named_modules()`，限定 `isinstance(module, torch.nn.Linear)`，用相同 matcher 统计目标/排除数量并抽查真实名字。不要把一份 Qwen 60 层模型的固定计数推广到全部变体；不要把量化的层匹配规则当作 PEFT LoRA target 的规则。

可对照当前存在的官方 shell（均相对框架根目录）：

- `examples/qwen_image/model_training/special/quant_training/Qwen-Image-2512-bitsandbytes_nf4.sh`：DiT NF4，排除调制/输入输出/时间嵌入的实际 Linear。
- `examples/z_image/model_training/special/quant_training/Z-Image-Turbo-bitsandbytes_nf4.sh`：DiT 与 text encoder 都 NF4，训练仍是 LoRA，不是两个模型全量更新。
- `examples/minimax_h3/model_training/special/quant_training/MiniMax-H3-FL2VA-bitsandbytes_nf4.sh`：核对实际 repo 字符串、视频/音频条件与 final/embed 层排除。

## 在线量化与预量化 checkpoint

已注册 `quant_config` 的 checkpoint 由 `ModelPool.resolve_quant_config()` 使用注册配置加载；传入的 quantize 选项被忽略并警告，不能覆盖其格式或在 packed 权重上再次在线量化。未注册的自定义预量化文件不一定能识别模型类型，也不能指望该保护自动生效；需要兼容的模型识别及与其保存配置一致的 `load_prequantized=true`。不为完成普通用户训练自动修改框架注册表。

在线训练记录底模 revision、method、mode、target/exclude、backend kwargs/版本和实际量化层；预量化训练记录精确 checkpoint 与注册/自定义配置。LoRA 导出主要是浮点 A/B 参数，不携带完整基座。推理应重建相同基座，在线量化使用与训练相同设置；仅更换为相同方法名的另一个预量化文件不保证误差一致。

## 验证/推理与显存管理

当前 loader 有 `quantize is not None` 且 **module_map=None** 的独立加载分支，支持不显式给 `vram_config` 的量化；不能再声称“无 vram_config 必然量化失效”。当前 `BasePipeline.load_lora()` 检测到 `quantize_config` 后会启用 LoRA hot-loading，避免把 LoRA 融入 packed 基座；仍应验证补丁数与生成结果。

验证时显存不足，可按目标官方低显存推理方式为 ModelConfig 配置四阶段 offload/onload/preparing/computation dtype/device，并使用实际测量后的 `vram_limit`。那是**推理**资源管理，不自动等同训练 hook offload；量化存储由 backend 管理，VRAM dtype 不会把 packed 权重简单转换成 BF16。不要为所有模型固定减 0.5/2 GiB 就宣称足够。

`offload_dtype` 和 `offload_device` 非 None 才启用 ModelPool 的 module map 分支；在线量化与 disk offload 的同组件组合被 loader 明确拒绝，disk 场景需要预量化权重。`--offload_models` 会生成 disk 配置，不能把它用作量化训练时的 CPU offload 替代开关。

最终验收覆盖：真实量化/排除命中、底模冻结、LoRA 可训练、有限输入梯度与更新、保存/重载、训练前后同条件效果，以及加载/训练/保存各阶段峰值；后端声明可微和预检成功都不是完整训练证明。
