# YuE2

YuE2 是一个统一了符号规划与音频生成的音乐生成模型。给定歌词与风格提示，模型先生成可编辑的旋律与和弦乐谱，再将乐谱渲染为包含人声与伴奏的 48kHz 立体声完整歌曲。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e ".[yue2,audio]"
```

`yue2` 额外依赖包含 `tiktoken` 分词器，`audio` 包含音频工具依赖。更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [m-a-p/YuE2-3B](https://www.modelscope.cn/models/m-a-p/YuE2-3B) 和 [m-a-p/YuE2-Vae](https://www.modelscope.cn/models/m-a-p/YuE2-Vae) 模型并进行推理。AR/NAR 主干已启用显存管理，框架会自动根据剩余显存控制模型参数的加载。

```python
from diffsynth.pipelines.yue2 import YuE2Pipeline, ModelConfig
from diffsynth.utils.data.audio import save_audio
import torch

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = YuE2Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="m-a-p/YuE2-3B", origin_file_pattern="model.safetensors", **vram_config),
        ModelConfig(model_id="m-a-p/YuE2-Vae", origin_file_pattern="model.safetensors", computation_dtype=torch.float32),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = (
    "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, "
    "lyrical memorable melody, unhurried phrasing, 88 BPM"
)
lyrics = (
    "[Verse]\n"
    "Neon fades along the lane\n"
    "Footsteps keep the time of rain\n"
    "[Chorus]\n"
    "Let the day come into view\n"
    "Every road begins with you"
)
audio = pipe(prompt=prompt, lyrics=lyrics, cot="full", seed=831001)
save_audio(audio, 48000, "YuE2.wav")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[m-a-p/YuE2-3B](https://www.modelscope.cn/models/m-a-p/YuE2-3B)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/yue2/model_inference/YuE2.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/yue2/model_inference_low_vram/YuE2.py)|—|—|—|—|
|[m-a-p/YuE2-3B: Edit](https://www.modelscope.cn/models/m-a-p/YuE2-3B)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/yue2/model_inference/YuE2-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/yue2/model_inference_low_vram/YuE2-Edit.py)|—|—|—|—|

## 模型推理

模型通过 `YuE2Pipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`YuE2Pipeline` 推理的输入参数包括：

* `prompt`: 风格描述，包括语言、流派、乐器、人声特征与速度。
* `lyrics`: 歌词文本，支持 `[Verse]`、`[Chorus]` 等段落标签。
* `cot`: 规划模式，可选值包括 `"full"`（带和弦符号的旋律）、`"melody"`（仅旋律，推荐用于翻唱）和 `"off"`（不使用符号规划），默认为 `"full"`。
* `abc`: 直接使用的外部 ABC 乐谱，要求 `cot="full"` 或 `cot="melody"`。
* `cfg_scale`: 分类器无条件引导比例，`full`/`melody` 默认为 1.0，`off` 默认为 1.01。
* `seed`: 随机种子。
* `num_inference_steps`: 声学阶段流匹配 ODE 迭代步数，默认为 30。
* `vae_core_frames`: VAE 解码分块的核心帧数，默认使用解码器配置中的值。
* `progress_bar_cmd`: 进度条。自回归阶段显示 token/s，声学阶段显示 steps/s。

模型分三阶段生成：自回归乐谱规划器生成可编辑的 ABC 乐谱，自回归语义生成器基于乐谱产出离散音乐 token，NAR 流匹配模型通过 midpoint ODE 从这些 token 合成 VAE 隐变量，最后由 VAE 解码为 48kHz 立体声波形。

`YuE2Pipeline.plan` 可导出用于编辑的符号乐谱。“规划-编辑-渲染”流程会先生成乐谱，再修改乐谱（旋律、和弦、速度或结构），最后通过 `abc=...` 渲染修改后的作品；参见 [YuE2-Edit.py](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/yue2/model_inference/YuE2-Edit.py)。

若显存不足，请参考[显存管理](../Pipeline_Usage/Model_Inference.md#显存管理)。

## 模型训练

YuE2 暂不支持训练。
