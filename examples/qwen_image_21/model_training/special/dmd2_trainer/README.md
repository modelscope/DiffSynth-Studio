# Qwen-Image 2.1 DMD2 LoRA

与 MOSS trainer 一样，只继承通用 `DMD2TrainingModule`，在自己的 `__init__` 中通过框架加载模型并添加双 LoRA。`get_pipeline_inputs` 沿用 Qwen-Image 2.1 SFT 的写法，`prepare_batch` 运行原生 pipeline units，返回 `inputs_shared, inputs_posi, inputs_nega`；预测、损失与交替优化都使用公共 trainer。冻结 DiT 主干，student 与 fake-score 分别训练独立 LoRA，可选 GAN；使用普通 Euler 轨迹。

在仓库根目录分别运行：

```bash
PYTHONPATH="./" bash examples/qwen_image_21/model_training/special/dmd2_trainer/train_Qwen-Image-2.1-8steps_lora.sh
PYTHONPATH="./" bash examples/qwen_image_21/model_training/special/dmd2_trainer/train_Qwen-Image-2.1-Turbo-3steps_lora.sh
```

脚本通过 ModelConfig 下载并加载对应的 Qwen DiT，共享 Qwen-Image-2.1 的 text encoder、VAE 和 processor。Base 和 Turbo 的 teacher CFG 均为 1，负提示词为单个空格 `negative_prompt=" "`，与原生 pipeline 默认值一致。Base 默认八步，Turbo 默认三步；两者的时间表均由原生 Qwen scheduler 在 1024×1024 参考分辨率下生成（`dynamic_shift_len=4096`）。

Base 八步时间表与已有 Qwen 八步 OPD 示例一致：

```text
sigmas:    1.0, 0.9160239696502686, 0.8200457692146301, 0.7092946171760559, 0.5800749063491821, 0.42734700441360474, 0.24405395984649658, 0.019999980926513672
timesteps: 1000.0, 916.0239868164062, 820.0457763671875, 709.2946166992188, 580.0748901367188, 427.3470153808594, 244.053955078125, 19.999980926513672
```

Turbo 三步时间表：

```text
sigmas:    1.0, 0.6080472469329834, 0.020000040531158447
timesteps: 1000.0, 608.0472412109375, 20.00004005432129
```

启动脚本开头使用公共工具计算时间表，并打印可填入脚本的 DMD2 参数：

```bash
PYTHONPATH="./" python examples/dev_tools/compute_dmd2_sigmas.py --scheduler_name "Qwen-Image" --num_inference_steps 8 --extra_kwargs '{"dynamic_shift_len": 4096}'
PYTHONPATH="./" python examples/dev_tools/compute_dmd2_sigmas.py --scheduler_name "Qwen-Image" --num_inference_steps 3 --extra_kwargs '{"dynamic_shift_len": 4096}'
```

最后一步推进到 sigma 0。训练和验证均显式使用对应模型的固定序列，不随当前图片尺寸重新计算动态 shift。推理加载对应底模与学生 LoRA 后，显式传入同一序列。以 Base 八步为例：

```python
image = pipe(prompt=prompt, cfg_scale=1, num_inference_steps=8, sigmas=[1.0, 0.9160239696502686, 0.8200457692146301, 0.7092946171760559, 0.5800749063491821, 0.42734700441360474, 0.24405395984649658, 0.019999980926513672])
```

`validate_lora.py` 根据 `model_id` 选择 Base 八步或 Turbo 三步固定 sigmas，并使用 `len(sigmas)` 作为推理步数，以 BF16 权重推理。先编辑脚本顶部的 `model_id`、`lora_checkpoint`、`prompt`、`height`、`width`、`seed` 和 `output_path`，再直接运行：

```bash
PYTHONPATH="./" python examples/qwen_image_21/model_training/special/dmd2_trainer/validate_lora.py
```

`model_id` 选择对应的 Base 或 Turbo 底模。验证默认 1024×1024，输出默认写入 `workspace/` 下带 UTC 时间戳的目录；复跑同一次验证时，将顶部 `output_path` 改为已有目录。

示例通过 ModelScope 下载 `DiffSynth-Studio/diffsynth_example_dataset` 中的 `qwen_image_21/Qwen-Image-2.1/*`，读取 `data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1/metadata.csv`，图像 base 路径为同目录。最大像素 1,048,576，数据重复 50 次、训练 5 epochs；复用框架的 RGBA 转换及 32 对齐预处理。

每卡 batch 1、梯度累积 2，进程数由 Accelerate 配置决定，有效 batch 为进程数乘以 2。学习率固定 `5e-6`，student/fake LoRA rank 与 alpha 均为 128，8 workers。

GAN 默认开启，在训练脚本中添加 `--disable_gan` 可关闭。关闭时图像只用于经过预处理后的尺寸，不编码真实图像。GAN 使用 `transformer_blocks.16` 的输出 hook，截取末尾目标图像 tokens 并还原为 BCHW，输入通道数从 DiT 的 `inner_dim` 获取。

训练关闭 KV cache，避免 student、teacher 和 fake 切换时复用缓存。teacher、student 和 fake 默认均使用 CFG 1，此时原生 units 不额外编码负提示词，预测也不执行负向分支；显式提高 teacher CFG 时，原生 units 会编码上述负提示词。LoRA 参数默认与 pipeline 一样使用 BF16，可通过 `--lora_dtype fp32` 设置两套 LoRA 为 FP32；VAE 保留 Qwen 原生 BF16 精度。

`--save_steps 100` 按 optimizer 实际更新次数计数，每完成一个梯度累积窗口计 1，每 100 次更新保存学生 LoRA。Base 和 Turbo 示例分别将训练参数与学生 LoRA 保存到 `./models/train/Qwen-Image-2.1_dmd2_lora_8steps` 和 `./models/train/Qwen-Image-2.1-Turbo_dmd2_lora_3steps`。训练配置直接修改脚本中的对应参数。
