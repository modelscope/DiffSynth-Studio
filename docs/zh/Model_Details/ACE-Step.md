# ACE-Step

ACE-Step 1.5 是一个开源音乐生成模型，基于 DiT 架构，支持文生音乐、音频翻唱、局部重绘等多种功能，可在消费级硬件上高效运行。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [ACE-Step/Ace-Step1.5](https://www.modelscope.cn/models/ACE-Step/Ace-Step1.5) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 3G 显存即可运行。

```python
from diffsynth.pipelines.ace_step import AceStepPipeline, ModelConfig
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


pipe = AceStepPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="acestep-v15-turbo/model.safetensors", **vram_config),
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/model.safetensors", **vram_config),
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    text_tokenizer_config=ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = "An explosive, high-energy pop-rock track with a strong anime theme song feel. The song kicks off with a catchy, synthesized brass fanfare over a driving rock beat with punchy drums and a solid bassline. A powerful, clear male vocal enters with a theatrical and energetic delivery, soaring through the verses and hitting powerful high notes in the chorus. The arrangement is dense and dynamic, featuring rhythmic electric guitar chords, brief instrumental breaks with synth flourishes, and a consistent, danceable groove throughout. The overall mood is triumphant, adventurous, and exhilarating."
lyrics = '[Intro - Synth Brass Fanfare]\n\n[Verse 1]\n黑夜里的风吹过耳畔\n甜蜜时光转瞬即万\n脚步飘摇在星光上\n心追节奏心跳狂乱\n耳边传来电吉他呼唤\n手指轻触碰点流点燃\n梦在云端任它蔓延\n疯狂跳跃自由无间\n\n[Chorus]\n心电感应在震动间\n拥抱未来勇敢冒险\n那旋律在心中无限\n世界变得如此耀眼\n\n[Instrumental Break - Synth Brass Melody]\n\n[Verse 2]\n鼓点撞击黑夜的底端\n跳动节拍连接你我俩\n在这里让灵魂发光\n燃尽所有不留遗憾\n\n[Instrumental Break - Synth Brass Melody]\n\n[Bridge]\n光影交错彼此的视线\n霓虹之下夜空的蔚蓝\n月光洒下温热心田\n追逐梦想它不会遥远\n\n[Chorus]\n心电感应在震动间\n拥抱未来勇敢冒险\n那旋律在心中无限\n世界变得如此耀眼\n\n[Outro - Instrumental with Synth Brass Melody]\n[Song ends abruptly]'
audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    duration=160,
    bpm=100,
    keyscale="B minor",
    timesignature="4",
    vocal_language="zh",
    seed=42,
)

save_audio(audio, pipe.vae.sampling_rate, "acestep-v15-turbo.wav")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[ACE-Step/Ace-Step1.5](https://www.modelscope.cn/models/ACE-Step/Ace-Step1.5)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/Ace-Step1.5.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/Ace-Step1.5.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/Ace-Step1.5.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/Ace-Step1.5.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/Ace-Step1.5.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/Ace-Step1.5.py)|
|[ACE-Step/acestep-v15-turbo-shift1](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-shift1)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-turbo-shift1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-shift1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-turbo-shift1.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-turbo-shift1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-turbo-shift1.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-shift1.py)|
|[ACE-Step/acestep-v15-turbo-shift3](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-shift3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-turbo-shift3.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-shift3.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-turbo-shift3.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-turbo-shift3.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-turbo-shift3.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-shift3.py)|
|[ACE-Step/acestep-v15-turbo-continuous](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-continuous)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-turbo-continuous.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-continuous.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-turbo-continuous.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-turbo-continuous.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-turbo-continuous.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-continuous.py)|
|[ACE-Step/acestep-v15-base](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-base.py)|
|[ACE-Step/acestep-v15-base: CoverTask](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-base-CoverTask.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-base-CoverTask.py)|—|—|—|—|
|[ACE-Step/acestep-v15-base: RepaintTask](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-base-RepaintTask.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-base-RepaintTask.py)|—|—|—|—|
|[ACE-Step/acestep-v15-sft](https://www.modelscope.cn/models/ACE-Step/acestep-v15-sft)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-sft.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-sft.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-sft.py)|
|[ACE-Step/acestep-v15-xl-base](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-xl-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-xl-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-xl-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-xl-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-xl-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-xl-base.py)|
|[ACE-Step/acestep-v15-xl-sft](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-sft)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-xl-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-xl-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-xl-sft.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-xl-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-xl-sft.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-xl-sft.py)|
|[ACE-Step/acestep-v15-xl-turbo](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-turbo)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-xl-turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-xl-turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-xl-turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-xl-turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-xl-turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-xl-turbo.py)|

## 模型推理

模型通过 `AceStepPipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`AceStepPipeline` 推理的输入参数包括：

* `prompt`: 音乐文本描述。
* `cfg_scale`: 分类器无条件引导比例，默认为 1.0。
* `lyrics`: 歌词文本。
* `task_type`: 任务类型，可选值包括 `"text2music"`（文生音乐）、`"cover"`（音频翻唱）、`"repaint"`（局部重绘），默认为 `"text2music"`。
* `reference_audios`: 参考音频列表（Tensor 列表），用于提供音色参考。
* `src_audio`: 源音频（Tensor），用于 cover 或 repaint 任务。
* `denoising_strength`: 降噪强度，控制输出受源音频的影响程度，默认为 1.0。
* `audio_cover_strength`: 音频翻唱步数比例，控制 cover 任务中前多少步使用翻唱条件，默认为 1.0。
* `audio_code_string`: 输入音频码字符串，用于 cover 任务中直接传入离散音频码。
* `repainting_ranges`: 重绘时间区间（浮点元组列表，单位为秒），用于 repaint 任务。
* `repainting_strength`: 重绘强度，控制重绘区域的变化程度，默认为 1.0。
* `duration`: 音频时长（秒），默认为 60。
* `bpm`: 每分钟节拍数，默认为 100。
* `keyscale`: 音阶调式，默认为 "B minor"。
* `timesignature`: 拍号，默认为 "4"。
* `vocal_language`: 演唱语言，默认为 "unknown"。
* `seed`: 随机种子。
* `rand_device`: 噪声生成设备，默认为 "cpu"。
* `num_inference_steps`: 推理步数，默认为 8。
* `shift`: 调度器时间偏移参数，默认为 1.0。

## 模型训练

ace_step 系列模型统一通过 `examples/ace_step/model_training/train.py` 进行训练，脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloader 的进程数量。
        * `--data_file_keys`: 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 `,` 分隔。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID。用逗号分隔。
        * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，以 `,` 分隔。
        * `--fp8_models`: 以 FP8 格式加载的模型，目前仅支持参数不被梯度更新的模型。
    * 训练基础配置
        * `--learning_rate`: 学习率。
        * `--num_epochs`: 轮数（Epoch）。
        * `--trainable_models`: 可训练的模型，例如 `dit`、`vae`、`text_encoder`。
        * `--find_unused_parameters`: DDP 训练中是否存在未使用的参数。
        * `--weight_decay`: 权重衰减大小。
        * `--task`: 训练任务，默认为 `sft`。
    * 输出配置
        * `--output_path`: 模型保存路径。
        * `--remove_prefix_in_ckpt`: 在模型文件的 state dict 中移除前缀。
        * `--save_steps`: 保存模型的训练步数间隔。
    * LoRA 配置
        * `--lora_base_model`: LoRA 添加到哪个模型上。
        * `--lora_target_modules`: LoRA 添加到哪些层上。
        * `--lora_rank`: LoRA 的秩（Rank）。
        * `--lora_checkpoint`: LoRA 检查点的路径。
        * `--preset_lora_path`: 预置 LoRA 检查点路径，用于 LoRA 差分训练。
        * `--preset_lora_model`: 预置 LoRA 融入的模型，例如 `dit`。
    * 梯度配置
        * `--use_gradient_checkpointing`: 是否启用 gradient checkpointing。
        * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。
        * `--gradient_accumulation_steps`: 梯度累积步数。
    * 分辨率配置
        * `--height`: 图像/视频的高度。留空启用动态分辨率。
        * `--width`: 图像/视频的宽度。留空启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
        * `--num_frames`: 视频的帧数（仅视频生成模型）。
* ACE-Step 专有参数
    * `--tokenizer_path`: Tokenizer 路径，格式为 model_id:origin_pattern。
    * `--silence_latent_path`: 静音隐变量路径，格式为 model_id:origin_pattern。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型。

### 样例数据集

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
