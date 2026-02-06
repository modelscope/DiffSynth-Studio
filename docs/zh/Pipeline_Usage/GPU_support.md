# GPU/NPU 支持

`DiffSynth-Studio` 支持多种 GPU/NPU，本文介绍如何在这些设备上运行模型推理和训练。

在开始前，请参考[安装依赖](/docs/zh/Pipeline_Usage/Setup.md)安装好 GPU/NPU 相关的依赖包。

## NVIDIA GPU

本项目提供的所有样例代码默认支持 NVIDIA GPU，无需额外修改。

## AMD GPU

AMD 提供了基于 ROCm 的 torch 包，所以大多数模型无需修改代码即可运行，少数模型由于依赖特定的 cuda 指令无法运行。

## Ascend NPU
### 推理
使用 Ascend NPU 时，需把代码中的 `"cuda"` 改为 `"npu"`。

例如，Wan2.1-T2V-1.3B 的推理代码：

```diff
import torch
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core.device.npu_compatible_device import get_device_name

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
-   "preparing_device": "cuda",
+   "preparing_device": "npu",
    "computation_dtype": torch.bfloat16,
-   "computation_device": "cuda",
+   "computation_device": "npu",
}
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
-   device="cuda",
+   device="npu",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
-   vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
+   vram_limit=torch.npu.mem_get_info(get_device_name())[1] / (1024 ** 3) - 2,
)

video = pipe(
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
)
save_video(video, "video.mp4", fps=15, quality=5)
```

#### USP(Unified Sequence Parallel)
如果想要在NPU上使用该特性，请通过如下方式安装额外的第三方库：
```shell
pip install git+https://github.com/feifeibear/long-context-attention.git
pip install git+https://github.com/xdit-project/xDiT.git
```

### 训练
当前已为每类模型添加NPU的启动脚本样例，脚本存放在`examples/xxx/special/npu_training`目录下，例如 `examples/wanvideo/model_training/special/npu_training/Wan2.2-T2V-A14B-NPU.sh`。

在NPU训练脚本中，添加了可以优化性能的NPU特有环境变量，并针对特定模型开启了相关参数。

#### 环境变量
```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```
`expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征。

```shell
export CPU_AFFINITY_CONF=1
```
设置0或未设置: 表示不启用绑核功能

1: 表示开启粗粒度绑核

2: 表示开启细粒度绑核

#### 特定模型需要开启的参数
| 模型        | 参数 | 备注                |
|-----------|------|-------------------|
| Wan 14B系列 | --initialize_model_on_cpu | 14B模型需要在cpu上进行初始化 |
| Qwen-Image系列 | --initialize_model_on_cpu | 模型需要在cpu上进行初始化 |