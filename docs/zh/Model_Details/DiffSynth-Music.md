# DiffSynth-Music

DiffSynth-Music 是 DiffSynth-Studio 团队基于 ACE-Step-1.5 训练的可控生成音乐模型套件，模型的基础架构沿用 ACE-Step-1.5，增加了多个额外的模块，支持节拍、人声、伴奏、韵律和音色参考等多种控制方式。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多安装信息请参考 [依赖安装](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码将加载 [DiffSynth-Studio/DiffSynth-Music](https://www.modelscope.cn/models/DiffSynth-Studio/DiffSynth-Music) 模型进行推理。示例覆盖六种用法：原生音乐合成、节拍控制、人声控制、伴奏控制、韵律控制和音色参考。

```python
import torch, torchaudio
from diffsynth.pipelines.diffsynth_music import DiffSynthMusicPipeline, ModelConfig
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.core.data.operators import LoadMultiTrackAudio
from diffsynth.utils.music_tools import extract_prosody, generate_click
from modelscope import snapshot_download


pipe = DiffSynthMusicPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="transformer/model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="conditioner/model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="vae/model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="track_separator/model.safetensors", computation_dtype=torch.float32),
    ],
    tokenizer_config=ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="text_encoder/"),
)
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="template_control/"),
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="template_prosody/"),
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="template_reference/"),
    ],
)

lyrics = "[Intro]\n\n你的歌词..."
prompt = "An explosive, high-energy pop-rock track with a strong anime theme song feel."
snapshot_download("DiffSynth-Studio/DiffSynth-Music", allow_file_pattern="assets/*", local_dir="data")

# 原生音乐合成
audio = template(
    pipe,
    prompt=prompt, negative_prompt=pipe.default_negative_prompt,
    lyrics=lyrics,
    duration=240,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=50,
)
torchaudio.save("audio_1_output.mp3", audio, 48000)

# 节拍控制
beats = generate_click(120, duration=240)
audio = template(
    pipe,
    prompt=prompt, negative_prompt=pipe.default_negative_prompt,
    lyrics=lyrics,
    duration=240,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"model_id": 0, "audio": beats}],
    negative_template_inputs=[{"model_id": 0, "audio": beats * 0}],
)
torchaudio.save("audio_2_output.mp3", audio, 48000)

# 人声控制
audio = LoadMultiTrackAudio(division_factor=3840)("data/assets/audio_reference.mp3")
vocals = pipe.extract_track(audio, track="vocals")
audio = template(
    pipe,
    prompt=prompt, negative_prompt=pipe.default_negative_prompt,
    lyrics="",
    duration=vocals.shape[1] / 48000,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"model_id": 0, "audio": vocals}],
    negative_template_inputs=[{"model_id": 0, "audio": vocals}],
    target_audio=vocals, target_track="vocals",
)
torchaudio.save("audio_3_output.mp3", audio, 48000)

# 伴奏控制
audio = LoadMultiTrackAudio(division_factor=3840)("data/assets/audio_reference.mp3")
music = pipe.extract_track(audio, track=["drums", "bass", "other"])
audio = template(
    pipe,
    prompt=prompt, negative_prompt=pipe.default_negative_prompt,
    lyrics=lyrics,
    duration=music.shape[1] / 48000,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"model_id": 0, "audio": music}],
    negative_template_inputs=[{"model_id": 0, "audio": music}],
    target_audio=music, target_track=["drums", "bass", "other"],
)
torchaudio.save("audio_4_output.mp3", audio, 48000)

# 韵律控制
audio = LoadMultiTrackAudio(division_factor=3840)("data/assets/audio_reference.mp3")
vocals = pipe.extract_track(audio, track="vocals")
prosody = extract_prosody(vocals)
audio = template(
    pipe,
    prompt=prompt, negative_prompt=pipe.default_negative_prompt,
    lyrics=lyrics,
    duration=prosody.shape[1] / 48000,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"model_id": 1, "audio": prosody}],
    negative_template_inputs=[{"model_id": 1, "audio": prosody}],
)
torchaudio.save("audio_5_output.mp3", audio, 48000)

# 音色参考
audio = LoadMultiTrackAudio(division_factor=3840)("data/assets/audio_reference.mp3")
audio = template(
    pipe,
    prompt="Music", negative_prompt="", # 音色由参考音频控制
    lyrics=lyrics,
    duration=200,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=100,
    template_inputs=[{"model_id": 2, "audio": audio}],
)
torchaudio.save("audio_6_output.mp3", audio, 48000)
```

低显存推理版本见 [examples/diffsynth_music/model_inference_low_vram/DiffSynth-Music.py](/examples/diffsynth_music/model_inference_low_vram/DiffSynth-Music.py)。

## 模型概览

|模型 ID|推理|低显存推理|全量训练|全量训练验证|LoRA 训练|LoRA 训练验证|
|-|-|-|-|-|-|-|
| [DiffSynth-Studio/DiffSynth-Music](https://www.modelscope.cn/models/DiffSynth-Studio/DiffSynth-Music) | [code](/examples/diffsynth_music/model_inference/DiffSynth-Music.py) | [code](/examples/diffsynth_music/model_inference_low_vram/DiffSynth-Music.py) | [code](/examples/diffsynth_music/model_training/full/DiffSynth-Music.sh) | [code](/examples/diffsynth_music/model_training/validate_full/DiffSynth-Music.py) | - | - |

## 模型推理

通过 `DiffSynthMusicPipeline.from_pretrained` 加载模型，详见 [加载模型](../Pipeline_Usage/Model_Inference.md#loading-models)。

`DiffSynthMusicPipeline` 推理的主要参数包括：

* `prompt`：音乐描述，指定流派、情绪和风格。
* `negative_prompt`：无分类器引导的负向提示词。
* `lyrics`：歌词。`[Intro]`、`[Verse]` 等结构标签必须单独占一行。留空可生成纯音乐。
* `bpm`：每分钟节拍数。默认 100。
* `timesignature`：拍号。默认 "4"。
* `keyscale`：调性与音阶。默认 "B minor"。
* `input_audio`：可选的输入音频张量，与 `denoising_strength` 配合实现音频到音频生成。
* `duration`：生成音频的时长（秒）。
* `num_inference_steps`：流匹配步数。
* `cfg_scale`：无分类器引导强度。
* `seed`：随机种子。
* `tiled`：是否使用分块 VAE 解码。
* `kv_cache`：预计算的 KV 缓存，加速推理。
* `target_audio` / `target_track`：可选的音轨融合，将输出中的指定音轨替换为 `target_audio` 中对应音轨。

`TemplatePipeline` 提供三个控制适配器，通过 `template_inputs` 中的 `model_id` 指定：

* `model_id: 0` — template_control：节拍/人声/伴奏控制。生成音频跟随给定音频的节奏。配合 `target_audio` 可将目标音轨融合进输出。
* `model_id: 1` — template_prosody：生成人声跟随给定音频的基频与韵律。使用 `extract_prosody` 构建输入。
* `model_id: 2` — template_reference：输出音色由参考音频控制。

显存不足时请参考 [显存管理](../Pipeline_Usage/Model_Inference.md#vram-management)。

## 模型训练

模型支持对 DiT（trainable models: `dit`）在 `(prompt, lyrics, audio)` 数据上进行 LoRA 训练和全量训练。训练脚本见 [examples/diffsynth_music/model_training](/examples/diffsynth_music/model_training/)。
