---
name: diffsynth-style
description: >
  Normalize code style for newly integrated model code in DiffSynth-Studio. Use this skill after
  diffsynth-model-code or diffsynth-pipeline has written/modified files during a model integration.
  This skill applies only to Python files that were just created or modified as part of model or
  feature integration — whether it's a new pipeline, a newly copied model component, or a newly
  written inference script.
  SCOPE: Only for integrating a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke after any other development — bug fixes, refactoring, performance tuning, or general code
  changes — and do NOT use it as a general-purpose formatter.
---

# DiffSynth-Studio: 代码风格化

在模型代码接入或 Pipeline 创建完成后，对新建/修改的文件进行**纯视觉层面**的风格统一。**不改变任何运行时行为。**

## 适用范围

| 文件类型 | 路径 | 何时运行 |
|---------|------|---------|
| Pipeline 文件 | `diffsynth/pipelines/{series}.py` | 新建或修改 pipeline 后 |
| 模型文件 | `diffsynth/models/{component}.py` | 新建或修改模型组件后 |
| Model Config | `diffsynth/configs/model_configs.py` | 新增模型注册项后 |
| 训练/推理脚本 | `diffsynth/training/` 或 `examples/` 下的 `.py` | 新建脚本后 |
| 训练脚本（.sh）| `examples/{series}/model_training/` 下的 `.sh` | 新建或修改训练脚本后 |

## 风格原则

**核心目标：紧凑、美观、可读。能单行就单行，绝不无故换行。**

### 规则 1：推理脚本中 ModelConfig 必须一行写完

**这是最高优先级的规则。** 所有推理/测试脚本里的 `ModelConfig(...)` 调用必须写在一行内，无论参数多长：

```python
# ✅ ModelConfig 一行写完
model_configs=[
    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
]

# ❌ 不要拆成多行
model_configs=[
    ModelConfig(
        model_id="Qwen/Qwen-Image",
        origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
    ),
]
```

### 规则 2：Pipeline Dict 按参数组划分

Pipeline `__call__` 中的三个 dict（`inputs_posi`、`inputs_nega`、`inputs_shared`）**按参数组换行**，不要全部塞在一行也不要每个参数单独一行：

```python
# ✅ 按语义组划分，每组一行
inputs_posi = {
    "prompt": prompt, "negative_prompt": "", "cfg_scale": 1.0,
    "image_cfg_scale": 1.0, "control_image": control_image,
}
inputs_nega = {
    "prompt": negative_prompt,
}
inputs_shared = {
    "height": height, "width": width,
    "seed": seed, "num_inference_steps": num_steps,
}
```

**分组规则**：
- 同类参数放一行（如 prompt 相关、shape 相关、control 相关）
- 每行不超过 150 字符
- 短 dict（参数 < 3 个）可以一行写完

### 规则 3：`extra_kwargs` 必须一行写完

**无论多长，`extra_kwargs` 的值必须写在一行内，不能换行。** 这是 ModelConfig 字典中的特殊规则，适用于 `model_configs.py` 中的所有 `extra_kwargs` 字段。

```python
# ✅ 正确 —— extra_kwargs 无论多长，写一行
{
    "model_hash": "349723183fc063b2bfc10bb2835cf677",
    "model_name": "wan_video_dit",
    "model_class": "diffsynth.models.wan_video_dit.WanModel",
    "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
}

# ❌ 错误 —— extra_kwargs 不能拆成多行
{
    "model_hash": "349723183fc063b2bfc10bb2835cf677",
    "model_name": "wan_video_dit",
    "model_class": "diffsynth.models.wan_video_dit.WanModel",
    "extra_kwargs": {
        'has_image_input': True,
        'patch_size': [1, 2, 2],
        'in_dim': 48,
        'dim': 1536,
    }
}
```

**什么时候可以换行？**
- 函数定义签名（`def __call__(...)`）的参数：每个参数一行是 OK 的
- Pipeline dict：按参数组换行
- 字符串过长（>200 字符）需要隐式拼接
- 确实无法在单行内合理表达的超长 dict（超过 300 字符且包含嵌套结构的）
- **但 `extra_kwargs` 是例外 —— 无论多长，必须一行**

### 规则 4：函数签名按语义分组

`from_pretrained` 和 `__call__` 的参数用注释按语义分组：

```python
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        # Tokenizers
        tokenizer_config: ModelConfig = ModelConfig(model_id="path/to/tokenizer"),
        # Optional
        vram_limit: float = None,
    ):
```

```python
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        # Steps
        num_inference_steps: int = 8,
        # Progress bar
        progress_bar_cmd=tqdm,
    ):
```

### 规则 5：Import 紧凑组织

标准库 import 合并为紧凑行：

```python
# ✅ 紧凑
import torch, math, warnings
import numpy as np
from PIL import Image
from typing import Union, List, Optional
from einops import rearrange, repeat
from tqdm import tqdm

# ❌ 拆太散
import torch
import math
import warnings
import numpy as np
from PIL import Image
from typing import Union
from typing import List
from typing import Optional
from einops import rearrange
from einops import repeat
from tqdm import tqdm
```

规则：
- `import torch, math, warnings` 合并一行
- `from typing import ...` 合并一行
- `from einops import rearrange, repeat` 合并一行
- `from PIL import Image`、`import numpy as np` 各占一行（使用频率高，单独成行更易扫读）
- 相对导入各占一行，按 `core` → `diffusion` → `models` → `utils` 排序
- **禁止重复 import**
- **删除不必要的 import** — 文件中引用过的 symbol 才需要保留 import。如果某个 import 的名称在整个文件中没有被使用（包括被 `from xxx import yyy` 导入的 `yyy` 和 `import xxx` 导入的 `xxx`），一律删除。常见场景：
  - 从外部库复制代码时带入的多余 import（如 `from diffusers import xxx` 但我们用的是 DiffSynth）
  - 目标库原始代码中的 import，但在 DiffSynth 的上下文中未使用
  - 调试/测试阶段临时添加的 import（如 `import pdb`、`from line_profiler import profile`）
  - `from typing import ...` 中未使用的类型
  - 被重命名覆盖的原 import（如 `import numpy as np` 但文件中从未出现 `np`）
  - 只出现在注释或字符串中的 import 不算"使用"

### 规则 6：空行精简

```python
import torch, math          # import 块
from PIL import Image
from ..core import ModelConfig

                                # 1 空行分隔 import 和代码
class MyPipeline(BasePipeline):

    def __init__(self, ...):     # class 和第一个方法之间 1 空行
        ...

                                # 方法之间 1 空行
    def __call__(self, ...):
        ...


                                # 类之间 2 空行
class MyUnit(PipelineUnit):
    ...


                                # 模块级函数之间 2 空行
def model_fn_xxx(dit, ...):
    ...
```

**不要多余空行**。特别是：
- 属性声明之间不加空行
- `# Parameters` 块内 dict 定义之间不加空行
- 方法内部连续语句之间不加空行
- 只在**逻辑块之间**加 1 个空行（如 `__call__` 的四个阶段之间）

### 规则 7：注释精简

- `__call__` 方法体用四阶段注释分隔：`# Scheduler` / `# Parameters` / `# Denoise` / `# Decode`
- 不要在短方法（<10 行）前写 docstring
- 模型文件的 `forward` 方法**不写 docstring**
- 行内注释：`# text`，紧跟代码后

```python
# ✅ __call__ 四阶段注释
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Parameters
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {"height": height, "width": width, "seed": seed}
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        ...

        # Decode
        self.load_models_to_device(['vae_decoder'])
        ...
```

### 规则 8：去除中文注释

**所有中文注释必须直接删除，不做翻译。** 代码库统一使用英文注释。

```python
# ❌ 中文注释
# 初始化管线
pipe = MyPipeline(device=device, torch_dtype=torch_dtype)

# ✅ 直接删除
pipe = MyPipeline(device=device, torch_dtype=torch_dtype)
```

```python
# ❌ 中文注释
# 加载模型到设备
self.load_models_to_device(['vae_decoder'])

# ✅ 直接删除（方法名已表达意图）
self.load_models_to_device(['vae_decoder'])
```

**判断标准**：只要包含中文字符，一律删除。不要翻译为英文，不要保留。如果注释内容确实重要（如算法逻辑说明），由人工后续补充英文。

### 规则 9：引号和 f-string

- 默认单引号：`'value'`
- 特殊值/格式字符串用双引号：`"cpu"`, `"B C H W"`, `f"layer_{idx}"`
- f-string 统一，不用 `.format()` 或 `%` 拼接

### 规则 10：属性声明紧凑排列

```python
# ✅ 紧凑排列，相关属性之间无额外空行
        self.scheduler = FlowMatchScheduler("Wan")
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.in_iteration_models = ("dit",)
        self.units = [WanUnit_ShapeChecker(), WanUnit_PromptEmbedder()]
        self.model_fn = model_fn_wan_video
```

### 规则 11：`from_pretrained` 紧凑写法

```python
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="path/to/tokenizer"),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = MyPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("text_encoder")
        pipe.dit = model_pool.fetch_model("dit")

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
```

fetch_model 调用紧凑排列，每行一个，不额外空行。

### 规则 12：禁止全局路径

**新接入的代码中不应包含任何全局/绝对路径。** 所有路径必须使用相对路径或数据集标准路径。

**检测模式** — 以下均为全局路径，需要替换：
- `/home/xxx/...`、`/root/...`、`/mnt/...` 等 Linux 绝对路径
- `C:\Users\xxx\...`、`D:\...` 等 Windows 绝对路径
- `~/xxx` 等带 `~` 的用户目录路径
- 其他非项目内、非 ModelScope 的全局路径

**替换方案** — 全局路径应替换为 diffsynth 标准数据集路径：

```python
# ❌ 全局路径
image = Image.open('/path/to/data/test_image.jpg')
model_path = '/path/to/models/wan_t2v_14b.safetensors'
video_path = '/path/to/project/test_video.mp4'

# ✅ 替换为标准数据集路径
image = Image.open('data/diffsynth_example_dataset/{series}/{ModelName}/images/001.jpg')
model_path = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
video_path = 'data/diffsynth_example_dataset/{series}/{ModelName}/videos/001'
```

**数据集使用方式**：
- 所有样例数据集统一托管在 ModelScope 的 `DiffSynth-Studio/diffsynth_example_dataset`
- 下载：`modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "{series}/{dataset_name}/*" --local_dir ./data/diffsynth_example_dataset`
- 路径格式：`data/diffsynth_example_dataset/{series}/{dataset_name}/images/xxx.jpg` 或 `videos/xxx`
- 详细使用规范见 [dataset-guidelines.md](../diffsynth-pipeline-training/references/dataset-guidelines.md)

**如果无法确定对应的数据集路径**，提醒用户确认后再替换，不要随意猜测。

### 规则 13：移除 AI 接入痕迹

**所有 AI 辅助生成的中间痕迹必须在风格化时清理掉。** 代码应该是干净的、可直接合并到主仓库的状态。

**必须清除的内容：**

| 痕迹类型 | 示例 | 处理方式 |
|---------|------|---------|
| **目标库引用注释** | `# 参考目标库 inference.py:123`、`# 对应目标库第 45-60 行`、`# Corresponds to target library: __call__ L285-291`、`Corresponds to target library: ...`（无论中英文） | 直接删除 |
| **AI 调试注释** | `# AI 测试用，后续删除`、`# 临时验证代码`、`# TODO: AI 添加的调试` | 直接删除 |
| **AI 测试路径** | `test_path = "/tmp/ai_test_output/"`、`debug_output_dir = "./ai_debug"` | 直接删除相关代码 |
| **AI 定义的环境变量** | `os.environ["AI_DEBUG_MODE"] = "1"`、`DEBUG_PATH = "/path/to/ai_debug/"` | 直接删除 |
| **AI 临时的 print** | `print("AI Debug: latent shape =", latents.shape)`、`print("=== AI test output ===")` | 删除，保留有意义的业务 print（如 "ControlNet doesn't support multi-ControlNet"） |
| **AI 生成的临时测试代码** | `if __name__ == "__main__":` 下 AI 自行添加的测试推理块（非正式推理脚本） | 直接删除 |
| **AI 占位符** | `"TODO: AI will implement this later"`、`pass  # AI placeholder` | 删除或替换为有意义的代码 |

**必须保留的内容：**

| 内容类型 | 示例 | 原因 |
|---------|------|------|
| 有意义的业务 print | `print("Z-Image ControlNet doesn't support multi-ControlNet. Only one image will be used.")` | 这是正常的运行时警告 |
| NPU/硬件相关的 warning | `warnings.warn("Replacing RMSNorm with NPU fusion operators...")` | 这是正式的功能提示 |
| 正式的 TODO | `# TODO: remove it`（来自目标库原始代码） | 这是目标库自带的 TODO，不是 AI 添加的 |

**判断标准**：如果这段注释/代码是 AI 在接入过程中为了理解、验证、调试而临时添加的，一律删除。如果是目标库原始代码中自带的、或有实际业务价值的，保留。

### 规则 14：禁止装饰性分隔注释

**所有文件中都不需要 `# ===` 装饰性分隔注释来分隔代码区块。** 这类注释包括 `# ============================================================`、`# {ClassName}`、`# === ClassName ===` 等格式，在 Pipeline 文件、`model_configs.py`、`vram_management_module_maps.py` 以及其他任何 `.py` 文件中都应当删除。代码结构通过 class/def 定义和空行来组织即可，不需要额外的分隔线。

```python
# ❌ model_configs.py / vram_management_module_maps.py 中的装饰性注释
# ============================================================
# ERNIE-Image
# ============================================================
{
    "model_hash": "xxx",
    ...
}

# ❌ 区块标题
# ----- Qwen-Image Support -----
{
    "model_hash": "yyy",
    ...
}

# ❌ 来源标注注释（属于 AI 接入痕迹，见规则 13）
    Corresponds to target library: __call__ L285-291 (noise init) + L343-351 (VAE decode pre-processing)
# Corresponds to target library: ...

# ✅ model_configs.py / vram_management_module_maps.py：干净的数据结构，不加任何装饰注释
{
    "model_hash": "xxx",
    ...
},
{
    "model_hash": "yyy",
    ...
}

# ❌ Pipeline 文件中的分隔注释——应当删除
class ErnieImagePipeline(BasePipeline):
    ...

# ❌ 同样应当删除
class ErnieImageUnit_ShapeChecker(PipelineUnit):
    ...
```

**判断标准**：所有 `.py` 文件中，`# ===` 分隔线、`# ----- xxx -----` 区块标题，一律删除。代码通过 class/def 和空行自然分隔即可。

### 规则 15：路径与命名一致性（强制执行）

**此规则不可跳过、不可协商、不可延迟。** 路径与命名一致性是风格化的核心要求之一，不是可选项。发现不一致时必须修复，不允许以"功能正常"、"改动范围大"、"影响多个文件"、"以后统一改"等理由跳过。

**训练、推理、数据集脚本的路径和文件名必须使用统一的命名约定。** 同一功能的脚本和数据集归属于同一组路径，所有路径中的名称必须一致。

**核心原则：一个功能对应一组脚本和数据，路径、文件名、数据集名称必须统一。**

**命名约定**：

| 位置 | 路径格式 | 说明 | 示例 |
|------|---------|------|------|
| **训练脚本** | `examples/{series}/model_training/` | series 是简短类别名 | `examples/ernie_image/model_training/train.py` |
| **训练子脚本** | `examples/{series}/model_training/{type}/{FeatureName}.sh` | feature 名，见规则 18 | `examples/ernie_image/model_training/lora/ERNIE-Image.sh` |
| **验证脚本** | `examples/{series}/model_training/validate_{type}/{FeatureName}.py` | 见规则 18 | `examples/ernie_image/model_training/validate_lora/ERNIE-Image.py` |
| **推理脚本** | `examples/{series}/model_inference/{FeatureName}.py` | PascalCase + 连字符，见规则 18 | `examples/ernie_image/model_inference/ERNIE-Image.py` |
| **数据集路径** | `data/diffsynth_example_dataset/{series}/{FeatureName}/` | | `data/diffsynth_example_dataset/ernie_image/ERNIE-Image/` |

**关键说明**：
- **series**：是简短的类别名（如 `ernie_image`、`wan`、`qwen_image`），不带 feature 后缀。多个相关功能可以共享同一个 series（如 `ERNIE-Image` 和 `ERNIE-Image-Turbo` 都属于 `ernie_image` series）
- **FeatureName**：是具体功能的标识符（如 `ERNIE-Image`），与模型 ID 严格对齐，见规则 18
- 所有脚本和数据集中的 series 名称和 feature 名称必须互相匹配，不能一个用 `ernie_image` 一个用 `ernie_image_t2i`

**强制检查与修复流程**：
1. **扫描所有相关文件**：提取训练脚本、推理脚本、验证脚本、数据集中的 series 名称和 feature 名称
2. **逐项比对**：发现任何不一致，列出所有受影响的路径和文件名
3. **确定统一标准**：优先使用蓝图/计划中已确认的名称；若无，选择最短且语义清晰的版本
4. **向用户展示差异和修正方案**：列出"当前不一致" → "修复后一致"的对照表
5. **用户确认命名方案后立即执行**：用户只决定"用哪个名称"，不决定"要不要改"。命名一致性修复是强制性的，不允许跳过。

```python
# ❌ 不一致：训练脚本用 ernie_image，推理脚本用 ernie_image_t2i
# 训练脚本: examples/ernie_image/model_training/lora/ERNIE-Image.sh
# 推理脚本: examples/ernie_image_t2i/model_inference/ERNIE-Image.py
# 数据集: data/.../ernie_image_t2i/ERNIE-Image/

# ✅ 一致：统一使用 ernie_image 作为 series
# 训练脚本: examples/ernie_image/model_training/lora/ERNIE-Image.sh
# 推理脚本: examples/ernie_image/model_inference/ERNIE-Image.py
# 数据集: data/.../ernie_image/ERNIE-Image/
```

### 规则 16：训练脚本（.sh）参数一行一个

**.sh 训练脚本中的命令行参数，每个参数独占一行（`--flag value \`），不要将多个参数塞在同一行。** 这样可以让长参数列表清晰可读，也方便单独修改某一参数。

**为什么重要**：训练脚本的参数通常很长（如 `--model_id_with_origin_paths` 包含多个模型路径），如果一行放多个参数，当某个参数需要修改时，整行都会变成过长的混乱状态。一行一个参数是最小化 diff、最大化可读性的最优选择。

**适用范围**：所有 `.sh` 训练脚本中的 `accelerate launch` 和 `modelscope download` 等命令。

```bash
# ✅ 正确 —— 每个参数一行
accelerate launch examples/ltx2/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/ltx2/LTX-2.3-T2AV-splited \
  --dataset_metadata_path data/diffsynth_example_dataset/ltx2/LTX-2.3-T2AV-splited/metadata.csv \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 512 \
  --width 768 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2.3-T2AV-full-splited-cache" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# ❌ 错误 —— 多个参数挤在一行
accelerate launch examples/ltx2/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/ltx2/LTX-2.3-T2AV-splited \
  --dataset_metadata_path data/diffsynth_example_dataset/ltx2/LTX-2.3-T2AV-splited/metadata.csv \
  --data_file_keys "video,input_audio" --extra_inputs "input_audio" \
  --height 512 --width 768 --num_frames 121 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors" \
  --learning_rate 1e-5 --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." --output_path "./models/train/LTX2.3-T2AV-full-splited-cache" \
  --trainable_models "dit" --use_gradient_checkpointing \
  --task "sft:data_process"
```

**要点**：
- 缩进使用 2 个空格
- 续行符 `\` 前面保留一个空格
- 布尔标志类参数（如 `--use_gradient_checkpointing`）也独占一行
- 注释行（`# 加载 VAE + TextEncoder`）可以保留，与命令之间空一行分隔不同阶段

### 规则 17：训练脚本（.sh）保持正确换行，不合并行

**.sh 训练脚本中的多行命令必须保持原有的换行结构，不能将多个参数合并到同一行。** 这是对规则 16 的补充 —— 规则 16 规定了"每个参数一行"，本规则强调"不要破坏已有正确的换行"。

**核心原则：一个 `--flag value` 对应一行，续行符 `\` 不能省略，不能合并。**

```bash
# ✅ 正确 —— 每个参数独占一行，换行完整保留
accelerate launch examples/ltx2/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/ltx2/LTX-2.3-T2AV-splited \
  --dataset_metadata_path data/diffsynth_example_dataset/ltx2/LTX-2.3-T2AV-splited/metadata.csv \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 512 \
  --width 768 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2.3-T2AV-full-splited-cache" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# ❌ 错误 —— 多个参数被合并到同一行，破坏了换行结构
accelerate launch examples/joyai_image/model_training/train.py \
    --dataset_base_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/metadata.csv" \
    --max_pixels 1048576 --dataset_repeat 1 \
    --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:JoyAI-Image-Und/model*.safetensors" \
    --learning_rate 1e-4 --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." --output_path "./models/train/JoyAI-Image-Edit-split-cache" \
    --lora_base_model "dit" --lora_target_modules "img_attn_qkv,txt_attn_qkv" --lora_rank 32 \
    --use_gradient_checkpointing --find_unused_parameters \
    --data_file_keys "image,edit_image" \
    --extra_inputs "edit_image" \
    --task "sft:data_process"
```

**为什么重要**：训练脚本参数通常很长，合并行会导致：
1. 行长度爆炸，难以扫读
2. 修改单个参数时 diff 不清晰（一行内多个改动混在一起）
3. 容易遗漏续行符 `\` 导致命令截断

**检查要点**：
- 每一行只能有一个 `--flag` 开头（加上其值）
- 每行末尾必须有 `\` 续行符（最后一行除外）
- 不能出现 `--flag1 val1 --flag2 val2 \` 这种一行两个参数的情况

### 规则 18：脚本文件名严格与模型 ID 对齐

**脚本文件名必须与模型 ID（`ModelConfig` 中的 `model_id`）保持一致。** 如果同一目录下不存在多个脚本会冲突的情况，类似 `-T2I`、`-I2I` 等功能描述后缀应当删除，文件名直接反映模型名称即可。

**核心判断**：脚本文件名 = 模型 ID 的功能变体。如果一个 series 下只有一个功能脚本，文件名不需要额外标识符。

```bash
# ✅ 模型 ID 为 PaddlePaddle/ERNIE-Image，目录下只有这一个脚本
# 推理: examples/ernie_image/model_inference/ERNIE-Image.py
# 低显存: examples/ernie_image/model_inference_low_vram/ERNIE-Image.py
# 全量训练: examples/ernie_image/model_training/full/ERNIE-Image.sh
# LoRA 训练: examples/ernie_image/model_training/lora/ERNIE-Image.sh

# ✅ 模型 ID 为 PaddlePaddle/ERNIE-Image-Turbo，目录下只有这一个脚本
# 推理: examples/ernie_image/model_inference/ERNIE-Image-Turbo.py
# 低显存: examples/ernie_image/model_inference_low_vram/ERNIE-Image-Turbo.py

# ❌ 不要添加 -T2I 后缀（只有一个文生图脚本时）
# 推理: examples/ernie_image/model_inference/ERNIE-Image-T2I.py
# 全量训练: examples/ernie_image/model_training/full/ERNIE-Image-T2I.sh

# ❌ 不要用小写或混合大小写（必须与模型 ID casing 一致）
# 推理: examples/ernie_image/model_inference/ernie-image.py
# 推理: examples/ernie_image/model_inference/Ernie-Image.py
```

**何时需要保留功能后缀**：同一 series 下存在多个不同类型的脚本时，需要用后缀区分。例如同时存在 T2I（文生图）和 I2I（图生图）：

```bash
# ✅ 同系列多个功能，需要后缀区分
examples/ernie_image/model_inference/ERNIE-Image-T2I.py
examples/ernie_image/model_inference/ERNIE-Image-I2I.py
```

**命名规则总结**：
1. 文件名与 `model_id` 的最后一部分（模型名称）完全对齐，包括大小写
2. 单一功能不加 `-T2I` 等后缀，多功能才加
3. 文档中的 code 链接必须与实际文件名严格一致

---

## 常见反模式

### 反模式：dict 每个参数一行

```python
# ❌ Pipeline dict 不要每个参数单独一行
inputs_posi = {
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "cfg_scale": 1.0,
    "image_cfg_scale": 7.5,
}

# ✅ 按语义组划分行
inputs_posi = {
    "prompt": prompt, "negative_prompt": negative_prompt, "cfg_scale": 1.0,
    "image_cfg_scale": 7.5,
}

# ✅ 短 dict 直接一行
inputs_posi = {"prompt": prompt}
```

### 反模式：import 拆太散

```python
# ❌
from typing import Union
from typing import Optional
from typing import List

# ✅
from typing import Union, Optional, List
```

### 反模式：保留未使用的 import

```python
# ❌ 以下 import 在文件中从未被使用
import torch
import math
from PIL import Image
from diffusers import AutoPipelineForText2Image
import pdb

# ✅ 只保留实际使用的 import
import torch
from PIL import Image
```

**检查方法**：逐个 import 检查其名称是否在文件代码中被引用，未使用的直接删除。

### 反模式：属性声明之间加空行

```python
# ❌
self.dit: MyDiT = None

self.vae: MyVAE = None

# ✅
self.dit: MyDiT = None
self.vae: MyVAE = None
```

### 反模式：函数调用参数无必要换行

```python
# ❌ 参数不多，完全可以一行
pipe = MyPipeline(
    device=device,
    torch_dtype=torch_dtype,
)

# ✅
pipe = MyPipeline(device=device, torch_dtype=torch_dtype)
```

### 反模式：ModelConfig 拆成多行

```python
# ❌ 推理脚本中 ModelConfig 不要拆行
model_configs=[
    ModelConfig(
        model_id="Qwen/Qwen-Image",
        origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
    ),
]

# ✅ 一行写完
model_configs=[
    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
]
```

### 反模式：中文注释

```python
# ❌ 中文注释（一律删除）
# 初始化管线
# 加载模型
# 去噪循环

# ✅ 直接删除
pipe = MyPipeline(device=device, torch_dtype=torch_dtype)
```

### 反模式：全局路径

```python
# ❌ 硬编码的全局路径
image = Image.open('/path/to/data/test_image.jpg')
model_path = '/path/to/models/wan_t2v_14b.safetensors'
video_path = 'C:\\path\\to\\project\\test_video.mp4'

# ✅ 使用标准数据集相对路径
image = Image.open('data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-14B/videos/001')
model_path = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
```

### 反模式：AI 接入痕迹残留

```python
# ❌ AI 接入痕迹
# 参考目标库 inference.py 第 123 行
# AI 测试：检查 latent 形状是否正确
print(f"AI Debug: latent shape = {latents.shape}")
os.environ["AI_DEBUG"] = "1"
# TODO: AI 后续需要删除这个临时变量
test_output_path = "/tmp/ai_test/"
# Corresponds to target library: __call__ L285-291 (noise init) + L343-351 (VAE decode)
    Corresponds to target library: __call__ L285-291 (noise init) + L343-351 (VAE decode pre-processing)

# ✅ 清理后的干净代码
# （相关痕迹已全部删除）
```

### 反模式：路径与命名不一致

```python
# ❌ 同一个 feature，训练脚本和推理脚本的 series 名称不一致
# 训练脚本: examples/ernie_image/model_training/lora/Ernie-Image-T2I.sh
# 推理脚本: examples/ernie_image_t2i/model_inference/Ernie-Image-T2I.py
# 数据集: data/diffsynth_example_dataset/ernie_image_t2i/Ernie-Image-T2I/

# ✅ 统一使用 ernie_image 作为 series，Ernie-Image-T2I 作为 feature 名
# 训练脚本: examples/ernie_image/model_training/lora/Ernie-Image-T2I.sh
# 推理脚本: examples/ernie_image/model_inference/Ernie-Image-T2I.py
# 数据集: data/diffsynth_example_dataset/ernie_image/Ernie-Image-T2I/
```

### 反模式：model_configs.py 的 extra_kwargs 换行

```python
# ❌
"extra_kwargs": {
    'has_image_input': True,
    'patch_size': [1, 2, 2],
    'in_dim': 16,
    'dim': 1536,
    ...
}

# ✅ 无论多长，写一行
"extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
```

### 反模式：model_configs.py / vram_management_module_maps.py 中的装饰性注释

**所有 `.py` 文件中的 `# ===` 装饰性分隔注释都应当删除。**

```python
# ❌ model_configs.py / vram_management_module_maps.py 中的装饰性分隔线
# ============================================================
# ERNIE-Image
# ============================================================

# ❌ 区块标题
# ----- Qwen-Image Support -----

# ❌ 来源标注（属于 AI 接入痕迹）
    Corresponds to target library: __call__ L285-291 (noise init) + L343-351 (VAE decode pre-processing)

# ✅ model_configs.py / vram_management_module_maps.py：干净的配置文件，不加任何装饰注释
```

### 反模式：训练脚本（.sh）破坏换行结构 / 多个参数挤在一行

**.sh 训练脚本中破坏换行结构、合并参数到同一行是错误的。见规则 16 和规则 17。**

```bash
# ❌ 错误 —— 多个参数挤在一行，难以阅读和修改
accelerate launch examples/joyai_image/model_training/train.py \
    --dataset_base_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/metadata.csv" \
    --max_pixels 1048576 --dataset_repeat 1 \
    --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:JoyAI-Image-Und/model*.safetensors" \
    --learning_rate 1e-4 --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." --output_path "./models/train/JoyAI-Image-Edit-split-cache" \
    --lora_base_model "dit" --lora_target_modules "img_attn_qkv,txt_attn_qkv" --lora_rank 32 \
    --use_gradient_checkpointing --find_unused_parameters \
    --data_file_keys "image,edit_image" \
    --extra_inputs "edit_image" \
    --task "sft:data_process"

# ✅ 正确 —— 每个参数一行
accelerate launch examples/joyai_image/model_training/train.py \
    --dataset_base_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/metadata.csv" \
    --max_pixels 1048576 \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:JoyAI-Image-Und/model*.safetensors" \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "./models/train/JoyAI-Image-Edit-split-cache" \
    --lora_base_model "dit" \
    --lora_target_modules "img_attn_qkv,txt_attn_qkv" \
    --lora_rank 32 \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --data_file_keys "image,edit_image" \
    --extra_inputs "edit_image" \
    --task "sft:data_process"
```

---

## 执行步骤

### Step 0: 读取蓝图信息

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**每个 skill 执行的第一步，强制要求。** 从蓝图报告中读取 Python 运行环境信息和本 skill 必要的信息。

```bash
# 从 CLAUDE.md 或 config.yaml 获取模型名称
MODEL_NAME="{model-name}"
BLUEPRINT_PATH="packages/${MODEL_NAME}/.sisyphus/integration-blueprints/${MODEL_NAME}-blueprint.md"
```

**本 skill 必须从蓝图报告中读取的信息：**

| 蓝图信息 | 用途 |
|---------|------|
| 基本信息表中的 `Conda 环境名称` | 验证环境 |
| 模型组件清单 | 确定需要风格化的新增文件 |
| 接入类型 | new_series 需要风格化所有文件，version_upgrade 只需风格化修改的文件 |

如果蓝图报告不存在，向用户说明原因并中止。

> 📝 完成后：更新渐进式报告 → `skill_work_report/style-report.md`

### Step 1: 初始化执行日志目录

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**读取蓝图信息后，** 创建执行日志目录结构：

```bash
EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_style"
mkdir -p ${EXEC_LOG_DIR}/{outputs,scripts,checkpoints}
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/style-report.md`

### Step 2: 制定执行计划

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**在开始风格化前，先制定完整的执行计划，输出到 `packages/{model-name}/.sisyphus/plans/style-plan.md`。** 基于蓝图信息和本次要处理的文件，明确目标文件、应用规则、全局路径替换方案、AI 痕迹清单项和备份方案。

Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」，详见 Plan 模板章节。

**执行计划需要写入 Plan 文件，并向用户展示，等待用户明确确认后才能继续后续步骤。**

> ⚠️ **必须等用户确认 Plan 无误后才能继续。** 不要跳过确认环节。用户未确认前，不要执行任何文件修改操作。

> 📝 完成后：更新渐进式报告 → `skill_work_report/style-report.md`

### Step 3: 确定需要风格化的文件

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**优先使用用户指定的文件。** 按以下优先级确定文件列表：

1. **用户直接指定** — 用户在调用时提到了具体文件路径或文件列表
2. **用户提到模型/管线名称** — 根据名称自动查找相关文件，例如：
   - 用户说"风格化 ernie_image" → 查找 `examples/ernie_image/` 下的训练/推理脚本
   - 用户说"风格化 wan_t2v" → 查找 `examples/wan_t2v/` 下的训练/推理脚本
3. **用户未指定任何目标** — 询问用户是否要自动查找，可选项：
   - 查找最近被 `diffsynth-model-code` 或 `diffsynth-pipeline` 创建/修改的文件
   - 查找最近的训练脚本（`examples/` 下新文件）
   - 查找最近的推理脚本（`examples/` 下新文件）
   - 查找 `model_configs.py` 中新增的注册项

将找到的文件列表展示给用户。

> 📝 完成后：更新渐进式报告 → `skill_work_report/style-report.md`

### Step 4: 自动备份并确认

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**在向用户逐条确认之前，不修改任何文件。**

1. **提醒用户先 commit**：

   > ⚠️ 风格化将修改以下文件（只改格式不改逻辑）。**请先 commit 当前工作状态再继续。**
   >
   > 只有 commit 才是可靠的恢复点。

   等待用户确认已 commit 后再继续。

2. **逐条确认**，列出每个文件将要应用的规则：

   | 文件 | 将应用的规则 |
   |------|-------------|
   | 训练/推理脚本 `.py/.sh` | 规则1(ModelConfig一行)、规则5(Import紧凑)、规则6(空行精简)、规则8(去中文注释)、规则12(禁止全局路径)、规则13(AI痕迹清理)、规则15(路径一致性，强制) |
   | `configs/model_configs.py` | 规则3(extra_kwargs一行)、规则14(禁止装饰性注释) |
   | `configs/vram_management_module_maps.py` | 规则14(禁止装饰性注释) |
   | 其他修改过的文件 | 根据文件类型应用对应规则 |

   **规则 12（禁止全局路径）的额外提醒**：
   > 发现以下全局路径将被替换为标准数据集路径：
   > - `/path/to/data/test.jpg` → `data/diffsynth_example_dataset/{series}/{ModelName}/images/001.jpg`
   >
   > 请确认替换是否正确，或提供你期望的路径。

   **规则 13（AI 痕迹清理）的额外提醒**：
   > 发现以下 AI 接入痕迹将被清除：
   > - `# 参考目标库 xxx.py:123` → 删除
   > - `print("AI Debug: ...")` → 删除
   > - `os.environ["AI_DEBUG"]` → 删除
   >
   > 请确认清理范围是否正确。

   **规则 14（model_configs.py 禁止装饰性注释）的额外提醒**：
   > 发现 `model_configs.py` / `vram_management_module_maps.py` 中的以下装饰性注释将被删除：
   > - `# =========...========` 分隔线
   > - `# ----- xxx -----` 区块标题
   > - `Corresponds to target library: ...` 来源标注
   >
   > 请确认删除范围是否正确。

   **规则 15（路径与命名一致性，强制修复）的额外提醒**：
   > 检查以下路径中的 series 和 feature 名称是否一致：
   > - Pipeline 文件名、模型文件名、训练脚本路径、推理脚本路径、数据集路径
   > - 发现不一致时，列出所有受影响的路径，提出统一的命名方案
   > - 此规则不可跳过：用户只决定"用哪个名称"，不决定"要不要改"
   >
   > 请确认统一后的名称是否正确。

3. **等待用户确认**后，才进入 Step 4。如果用户不同意某项规则的应用，跳过该文件的对应规则。

   > ⚠️ **此处是最后的确认关口。** 所有风格化修改必须在用户逐项确认规则应用范围后才执行。不要跳过确认直接修改文件。
   >
   > ⛔ **规则 15（路径与命名一致性）不可跳过。** 用户只决定"用哪个名称"，不决定"要不要改"。如果存在命名不一致，必须修复。

> 📝 完成后：更新渐进式报告 → `skill_work_report/style-report.md`

### Step 5: 逐文件应用风格规则

> 📖 开始前：重读本步骤描述，确认流程与报告路径

对每个已确认的文件：
1. 读取文件全文
2. 按确认的规则逐项检查并修改
3. **路径一致性检查（规则 15，强制）**：确认脚本中的 feature 简写与 Pipeline 文件名、数据集路径一致；发现不一致时必须修复，不允许跳过
4. **只做纯格式修改**：换行、空行、引号、import 组织、注释、路径替换
5. 不改变任何变量名、函数名、类名、逻辑结构、算法

> 📝 完成后：更新渐进式报告 → `skill_work_report/style-report.md`

### Step 6: 验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

1. 确保文件能正常 `import`
2. 如有测试，运行测试确认输出不变
3. 报告修改了哪些文件，每项改了什么

> 📝 完成后：更新渐进式报告 → `skill_work_report/style-report.md`

### Step 7: 更新蓝图 — 补充测试脚本清单

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**风格化完成后，** 在蓝图报告中追加一个「测试脚本补充说明」章节，记录本次接入的推理、训练、数据集相关脚本，明确后续 `diffsynth-testing` skill 需要测试的范围。

**蓝图报告路径**：`packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`

**追加章节**（如已有则更新内容）：

```markdown
## 补充：推理脚本清单（供 diffsynth-testing 使用）

> 本章节由 diffsynth-style skill 在风格化完成后追加，用于指导后续 diffsynth-testing 的测试范围。
> 注意：训练相关测试由 diffsynth-pipeline-training skill 负责，不在 diffsynth-testing 范围内。

### 推理脚本（需测试）

| 脚本路径 | 功能 | 测试目标 |
|---------|------|---------|
| `examples/{series}/model_inference/{ModelName}.py` | 基础推理 | 能正常加载模型并生成输出 |
| `examples/{series}/model_inference/{VariantName}.py` | {功能描述} | {具体测试目标} |

### 训练脚本（记录，不在此测试）

> 以下脚本由 diffsynth-pipeline-training skill 在训练验证阶段测试，diffsynth-testing 不运行训练相关脚本。

| 脚本路径 | 训练类型 | 说明 |
|---------|---------|------|
| `examples/{series}/model_training/train.py` | 训练框架 | TrainingModule 能正常加载、forward 可运行 |
| `examples/{series}/model_training/lora/{ModelName}.sh` | LoRA 训练 | 启动正常，能完成至少 1 个 step |
| `examples/{series}/model_training/full/{ModelName}.sh` | 全量训练 | 启动正常，能完成至少 1 个 step |

### 数据集相关

| 数据类型 | 路径 | 说明 |
|---------|------|------|
| 推理样例 | `data/diffsynth_example_dataset/{series}/{ModelName}/` | 推理脚本使用的样例数据 |
| 训练样例 | `data/diffsynth_example_dataset/{series}/{ModelName}/metadata.csv` | 训练数据集元信息 |
```

**填充要求**：
- 只记录**实际存在**的脚本，不要填充不存在的条目
- 如果推理脚本不存在，向用户说明原因并中止
- 如果某类训练脚本不存在（如没有全量训练脚本），在对应表格下方注明：`**注**：本模型无 {xxx} 脚本`
- 「测试目标」一栏要具体明确，让后续执行 diffsynth-testing 的人知道测试的目的是什么
- 如果风格化过程中修改了脚本路径（如重命名），使用**修改后**的新路径
- 如果风格化过程中发现并删除了无效脚本，在对应位置注明：`已删除（原因：{原因}）`

> 📝 完成后：更新渐进式报告 → `skill_work_report/style-report.md`

### Step 8: 最终验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

在所有风格化步骤完成后，执行最终验证：

```bash
# 1. 检查三个报告文件是否存在
for f in \
  "packages/{model-name}/.sisyphus/plans/style-plan.md" \
  "packages/{model-name}/.sisyphus/skill_work_report/style-report.md" \
  "packages/{model-name}/.sisyphus/user_report/style-report.md"; do
  if [ ! -f "$f" ]; then
    echo "WARNING: 缺失报告文件: $f"
  fi
done

# 2. 检查所有执行的测试脚本是否存在于执行日志目录
EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_style"
for script in "$EXEC_LOG_DIR/scripts/"*; do
  if [ -f "$script" ]; then
    echo "OK: $script 已保存"
  else
    echo "WARNING: 测试脚本缺失: $script"
  fi
done
```

如有缺失，立即补充。

> 📝 完成后：更新渐进式报告 → `skill_work_report/style-report.md`

## 输出

### 执行日志

所有执行过程保存到：
- **执行日志目录**: `packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_style/`
- **脚本目录**: `scripts/` - 保存的验证脚本
- **输出目录**: `outputs/` - 命令执行日志、diff 对比结果

### Plan

在制定执行计划步骤，将详细执行计划输出到 `packages/{model-name}/.sisyphus/plans/style-plan.md`。Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」。模板如下：

```markdown
# Style 执行 Plan

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-style |
| 执行时间 | {timestamp} |
| 接入类型 | {new_series / version_upgrade} |

## 执行步骤规划

以下按顺序列出所有执行步骤。每个步骤包含：目标、具体执行内容、产出物、注意事项。

### Step 0: 读取蓝图信息

**目标**：从蓝图报告中获取风格化所需的上下文信息。

**执行内容**：
- 读取 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- 提取：Conda 环境名称、模型组件清单、接入类型
- 如果蓝图报告不存在，向用户说明原因并中止

**产出物**：确认蓝图信息可用

---

### Step 1: 初始化执行日志目录

**目标**：创建执行日志目录结构。

**执行内容**：
- 创建 `packages/{model-name}/.sisyphus/execution-logs/{timestamp}_style/` 目录及子目录

**产出物**：执行日志目录

---

### Step 2: 制定执行计划

**目标**：输出本 Plan 文件，向用户展示风格化规划并确认。

**执行内容**：
- 将本 Plan 内容输出到 `packages/{model-name}/.sisyphus/plans/style-plan.md`
- 向用户展示风格化文件清单、应用规则、备份方案
- 等待用户明确确认后才继续

**产出物**：
- `packages/{model-name}/.sisyphus/plans/style-plan.md`

**注意事项**：
- 必须等用户确认后才能执行任何文件修改

---

### Step 3: 确定需要风格化的文件

**目标**：确定本次风格化涉及的文件列表。

**执行内容**：
- 优先使用用户指定的文件
- 否则根据模型名称自动查找相关文件
- 如果用户未指定，自动查找最近修改的文件
- 展示文件列表给用户确认

**产出物**：确认后的风格化文件清单

---

### Step 4: 自动备份并确认

**目标**：提醒用户 commit 当前状态，确保风格化修改有可靠的恢复点。

**执行内容**：
- 提醒用户先 commit 当前工作状态
- 等待用户确认已 commit 后再继续

**产出物**：git commit 快照

---

### Step 5: 逐文件应用风格规则

**目标**：对每个文件应用风格规则，只改格式不改逻辑。

**执行内容**：
- 按文件类型应用对应规则：
  - 训练/推理脚本：ModelConfig 一行、import 紧凑、空行精简、去中文注释、全局路径替换、AI 痕迹清理、路径一致性
  - model_configs.py：extra_kwargs 一行
  - Pipeline 文件：import 紧凑、Dict 参数分组、分隔符规范
  - 模型文件：import 紧凑、extra_kwargs 一行
- 逐文件应用，每完成一个文件记录变更

**产出物**：风格化后的代码文件

**注意事项**：
- 不改变任何运行时行为
- 只改格式，不改逻辑

---

### Step 6: 验证

**目标**：验证风格化未破坏代码功能。

**执行内容**：
- import 验证：确保所有修改后的文件可以正常导入
- 功能验证：运行基础推理脚本验证功能正常

**产出物**：验证通过确认

---

### Step 7: 更新蓝图 — 补充测试脚本清单

**目标**：在蓝图报告中追加测试脚本清单，指导后续 testing。

**执行内容**：
- 在蓝图报告中追加本次新增/修改的训练/推理脚本路径
- 确保后续 diffsynth-testing 能找到所有需要测试的脚本

**产出物**：更新后的蓝图报告

---

### Step 8: 最终验证

**目标**：确认所有报告文件和执行脚本完整性。

**执行内容**：
- 检查三个报告文件是否存在：Plan 文件、skill_work_report、user_report
- 检查执行日志目录中的脚本完整性
- 如有缺失，立即补充

**产出物**：验证通过确认

---

## 风格化规划

### 风格化文件清单
- 本次涉及修改的所有文件（Pipeline、模型、训练脚本、推理脚本等）

### 应用规则
- ModelConfig 一行写完
- import 紧凑（一行合并、去重复）
- 去除中文注释和 AI 痕迹
- 空行精简
- 全局路径替换为数据集标准路径 `data/diffsynth_example_dataset/{series}/{ModelName}/...`（模型权重改用 `ModelConfig`）
- 路径与命名一致性（训练/推理/数据集统一 series 和 feature 名）

### 脚本文件命名一致性修复
- 训练脚本路径：`examples/{series}/model_training/...`
- 推理脚本路径：`examples/{series}/model_inference/{FeatureName}.py`
- 训练子脚本文件名：`{type}/{FeatureName}.sh`
- 数据集路径：`data/.../{series}/{FeatureName}/`
```

### 渐进式步骤报告

**每个步骤完成后立即追加记录**。格式详见 [step-report.md](../diffsynth-integrator/references/step-report.md)。

报告路径：`packages/{model-name}/.sisyphus/skill_work_report/style-report.md`

**步骤划分**（与上方「执行步骤」章节的 Step 0-8 一一对应）：

| Step | 名称 | 对应 Workflow |
|------|------|---------------|
| 0 | 读取蓝图信息 | Step 0 |
| 1 | 初始化执行日志目录 | Step 1 |
| 2 | 制定执行计划 | Step 2 |
| 3 | 确定需要风格化的文件 | Step 3 |
| 4 | 自动备份并确认 | Step 4 |
| 5 | 逐文件应用风格规则 | Step 5 |
| 6 | 验证 | Step 6 |
| 7 | 更新蓝图 | Step 7 |
| 8 | 最终验证 | Step 8 |

每完成一个步骤，执行：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/style-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败 / ⬜ 跳过
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

### 向用户报告

风格化完成后，向用户报告 **必须写入文件**：

```bash
cat > packages/{model-name}/.sisyphus/user_report/style-report.md << 'OUTER_EOF'
## ✅ 代码风格化完成

执行日志: `packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_style/`

### 📁 风格化的文件

| 文件 | 应用的规则 |
|------|-----------|
| {file1} | {rules applied} |
| {file2} | {rules applied} |

### 📋 主要改动

- ModelConfig 一行写完: {N} 处
- Import 紧凑化: {N} 处
- 中文注释删除: {N} 处
- 全局路径替换: {N} 处
- AI 痕迹清理: {N} 处
OUTER_EOF
```
