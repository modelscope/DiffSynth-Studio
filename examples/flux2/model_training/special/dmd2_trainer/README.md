# FLUX.2-klein-base-4B DMD2 LoRA

`train.py` 继承公共 `DMD2TrainingModule`，使用框架的模型加载、LoRA 注入及训练循环。`get_pipeline_inputs` 沿用 FLUX.2 SFT 的输入格式，`prepare_batch` 通过原生 pipeline units 编码文本、图像并初始化噪声；预测直接复用 `pipe.model_fn`。Teacher、student LoRA 和 fake LoRA 共用冻结主干，默认启用 GAN 并额外训练分类头。

在仓库根目录运行：

```bash
PYTHONPATH="./" bash examples/flux2/model_training/special/dmd2_trainer/train_FLUX.2-klein-base-4B-8steps_lora.sh
```

示例直接使用 `accelerate launch`，设备数量由 Accelerate 配置决定。每卡 micro-batch 1、梯度累积 2，有效 batch 为进程数乘以 2；rank 128、固定学习率 5e-6、5 epochs、8 workers。每完成 100 个梯度累积窗口保存 student LoRA；每个窗口更新 critic，generator 在第 1、6、11……个窗口更新。默认输出到 `./models/train/FLUX.2-klein-base-4B_dmd2_lora_8steps`。LoRA 目标使用现有 FLUX.2 SFT 示例的逗号分隔模块列表。需要调整训练配置时，直接修改脚本中的对应参数。

脚本下载标准 `DiffSynth-Studio/diffsynth_example_dataset` 的 `flux2/FLUX.2-klein-base-4B` 子集，从对应的 `metadata.csv` 读取 `prompt` 和 `image`，`dataset_repeat=50`。在训练脚本中添加 `--disable_gan` 可关闭 GAN，仍从图片取得尺寸，但不编码真实图像。

Teacher 使用 base 模型的真实 CFG 4；student 以 CFG 1 推理。`embedded_guidance=1` 与 SFT 一致；klein-base-4B 没有 guidance embedding。噪声和 latent 沿用原生 `[B, H*W, 128]` 格式，噪声 dtype 跟随 pipeline。GAN hook 默认取 `single_transformer_blocks.7` 中位于文本之后的图像 token，再恢复成特征图；也可选择 `transformer_blocks` 的双流 block，此时读取其输出 tuple 的图像分支。通道数由 `dit.inner_dim` 得到。

这份八步脚本固定训练 1024×1024，`--max_pixels 1048576`。步点由原生 FLUX.2 scheduler 在 `dynamic_shift_len=4096` 时生成：`[1.0, 0.9854254723, 0.9666411281, 0.9415147305, 0.9061826468, 0.8528416753, 0.7630144954, 0.5798068047]`，最后一步推进到零。FLUX.2 的默认时间表随分辨率变化，改变分辨率时须同时生成对应训练时间表。

脚本开头调用共用的 `examples/dev_tools/compute_dmd2_sigmas.py`，按 `dynamic_shift_len=4096` 打印 8 步学生时间表，输出可直接填入 `--dmd2_student_sigmas`、`--dmd2_student_timesteps`。

训练后使用原生 pipeline 的八步 Euler。先编辑 `validate.py` 顶部的 `lora_checkpoint`、`prompt`、`height`、`width`、`seed` 和 `output_path`，再直接运行：

```bash
PYTHONPATH="./" python examples/flux2/model_training/special/dmd2_trainer/validate.py
```

验证脚本使用 1024×1024、8 步、CFG 1，时间表与训练一致。输出默认写入 `workspace/` 下带 UTC 时间戳的目录；复跑同一次验证时，将顶部 `output_path` 改为已有目录。
