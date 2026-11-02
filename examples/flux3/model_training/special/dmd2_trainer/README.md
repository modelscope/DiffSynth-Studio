# FLUX.3 Image Klein Base DMD2 LoRA

按 MOSS trainer 的结构接入：只继承 `DMD2TrainingModule`，保留自己的模型初始化，复制 SFT 的 `get_pipeline_inputs`，通过原生 pipeline units 准备输入。预测复用公共 trainer 的 `pipe.model_fn`，没有改动推理 pipeline 或网络代码。

支持 `black-forest-labs/flux-3-image-klein-base` 的普通 Euler 蒸馏。`flux-3-image-klein` 的两时间条件 flow-map 主干不是这个 velocity 训练目标，入口会拒绝该底模；其 text encoder、processor 和 VAE 可供 base 模型复用。

在仓库根目录运行：

```bash
PYTHONPATH="./" bash examples/flux3/model_training/special/dmd2_trainer/train_flux-3-image-klein-base-8steps_lora.sh
```

默认八步，使用原生 FLUX.3 scheduler 的 `alpha=6.93`：

```text
sigmas:    1.0, 0.9798020720481873, 0.954107403755188, 0.920318603515625, 0.873896598815918, 0.8061264157295227, 0.6978851556777954, 0.4974874258041382
timesteps: 1000.0, 979.8020629882812, 954.107421875, 920.318603515625, 873.8966064453125, 806.1264038085938, 697.8851318359375, 497.4874267578125
```

最后一步推进到 sigma 0。该时间表不依赖图片分辨率；teacher CFG=3.5，student/fake CFG=1。示例直接使用 `accelerate launch`，设备数量由 Accelerate 配置决定；每卡 micro-batch 1、accumulation=2，有效 batch为进程数乘以2。student/fake LoRA rank128，alpha等于rank，固定学习率5e-6，5 epochs、workers8、最大像素1,048,576。两份LoRA默认BF16，可在命令中加 `--lora_dtype fp32` 改为FP32。底座冻结。

脚本开头调用共用的 `examples/dev_tools/compute_dmd2_sigmas.py`，按 `alpha=6.93` 打印 8 步学生时间表，输出可直接填入 `--dmd2_student_sigmas`、`--dmd2_student_timesteps`。

脚本下载标准 `DiffSynth-Studio/diffsynth_example_dataset` 的 `flux3/flux-3-image-klein-base` 子集，从对应的 `metadata.csv` 读取 `prompt`、`image`，`dataset_repeat=50`。默认启用GAN；在训练脚本添加 `--disable_gan` 可关闭，图像仍提供尺寸但不编码为真实样本。GAN在fake主干的 `single_blocks.13` 截取目标图像token，按图片尺寸恢复BCHW，分类头通道数从主干读取。

FLUX.3 原生 latent 是 `[image_tokens, 128]`，没有batch轴。子类只在两个地方补足图像粒度：`sample_noise` 让整张图共用一个随机sigma；`distribution_matching_loss` 按整张图的全部token与通道计算归一化尺度。其余rollout、teacher/fake、GAN loss和交替优化都使用公共实现。原生 `Packed` 条件仍由pipeline units构造，正负条件分别传递给原生model_fn。

每个累积窗口更新critic，generator在第1、6、11……个窗口更新。`save_steps=100` 表示每100次实际optimizer更新保存student LoRA。默认输出到 `./models/train/flux-3-image-klein-base_dmd2_lora_8steps`。LoRA 目标使用逗号分隔的 `q_proj,k_proj,v_proj,attn_out,mlp_in,mlp_out`，对应各 transformer block 的注意力与 MLP 线性层。

推理直接加载 base 底模与学生 LoRA，使用原生 pipeline 八步 Euler。先编辑 `validate.py` 顶部的 `lora_checkpoint`、`prompt`、`height`、`width`、`seed` 和 `output_path`，再直接运行：

```bash
PYTHONPATH="./" python examples/flux3/model_training/special/dmd2_trainer/validate.py
```

验证脚本默认八步、alpha 6.93、CFG 1、BF16 底模、1024×1024。输出默认写入 `workspace/` 下带 UTC 时间戳的目录；复跑同一次验证时，将顶部 `output_path` 改为已有目录。
