# MiniMax-Music3

MiniMax-Music3 是一个音乐生成模型，采用自回归语言模型与流匹配声学模型级联的两阶段架构，输入音乐描述与歌词即可生成带人声的立体声歌曲。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [MiniMax/MiniMax-Music3](https://www.modelscope.cn/models/MiniMax/MiniMax-Music3) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 6G 显存即可运行。

```python
from diffsynth.pipelines.minimax_music3 import MiniMaxMusic3Pipeline, ModelConfig
from diffsynth.utils.data.audio import save_audio
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = MiniMaxMusic3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="language_model/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="rvq_depth_decoder/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="condition_encoder/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="vocoder/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

lyrics = (
    "[verse]\n"
    "Morning light filtering through the pine\n"
    "Every quiet street is yours and mine\n"
    "[chorus]\n"
    "Softly the world begins to breathe"
)
prompt = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
    "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
    "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
)
audio = pipe(prompt=prompt, lyrics=lyrics, max_audio_duration=60.0, num_inference_steps=30, cfg_scale=1.7, seed=7)
save_audio(audio, 44100, "MiniMax-Music3.wav")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[MiniMax/MiniMax-Music3](https://www.modelscope.cn/models/MiniMax/MiniMax-Music3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_music3/model_inference/MiniMax-Music3.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_music3/model_inference_low_vram/MiniMax-Music3.py)|—|—|—|—|

## 模型推理

模型通过 `MiniMaxMusic3Pipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`MiniMaxMusic3Pipeline` 推理的输入参数包括：

* `prompt`: 音乐描述，用于指定风格、BPM、调性、人声特征与编曲。
* `lyrics`: 歌词。`[verse]`、`[chorus]` 等结构标签需各自独占一行，与标签同行的文本会被丢弃。留空时生成纯器乐。
* `max_audio_duration`: 生成音频时长的上限，单位为秒。自回归阶段可能提前结束，因此实际时长可能短于该值；帧数上限为 9000 帧。
* `num_inference_steps`: 每个窗口的流匹配迭代步数。
* `cfg_scale`: 声学阶段的 classifier-free guidance 强度。
* `seed`: 随机种子。
* `rand_device`: 随机数生成所在的设备。设为 `"cpu"` 可获得与计算设备无关的复现结果。
* `progress_bar_cmd`: 进度条。每个窗口显示一条覆盖全部迭代步的进度条。

模型分两阶段生成：自回归语言模型逐帧产出语义 token 与残差 RVQ 码，其逐帧隐状态作为条件，驱动分块流匹配模型生成 Flow-VAE 隐变量，最后由声码器合成 44.1kHz 立体声波形。自回归阶段的离散采样对数值精度敏感，因此该阶段的模型参数常驻显存，逐层显存管理仅作用于声码器。

若显存不足，请参考[显存管理](../Pipeline_Usage/Model_Inference.md#显存管理)。

## 模型训练

MiniMax-Music3 暂不支持训练。
