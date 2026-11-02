# MOSS Euler DMD2

使用普通 Euler 学生轨迹，以选中步骤的 `x_sigma - sigma * velocity` 计算分布匹配损失。teacher、student 和 fake-score 共用冻结的 DiT 主干：teacher 关闭 LoRA，student 使用 `default` LoRA，fake-score 使用独立的 `fake` LoRA；默认启用 GAN，可通过参数关闭。

主要代码：

- `train.py`：只继承公共 `DMD2TrainingModule`，将 SFT 的 `get_pipeline_inputs` 直接复制到本类。保留自己的构造函数，加载 MOSS pipeline、处理 VAE 和设置 DMD2 训练模式。实现 MOSS 的 `prepare_batch`、文本长度设置和 GAN 特征布局；main 保持独立。
- `examples/moss_image/model_training/train.py`：已有 SFT 入口，提供模型创建和原始数据到 pipeline 输入字典的转换。
- `diffsynth/diffusion/dmd2_trainer.py`：公共 `DMD2_parser`、双 LoRA 管理、Euler 轨迹、分类头、特征 hook 和单优化器训练循环。`forward` 按阶段分发，仅在 generator 分支中无梯度调用 `prepare_batch`；`forward_generator` 负责学生样本生成及 DM/GAN 损失，`forward_critic` 负责 fake 去噪及 GAN 真/假样本损失。两者接收同一份预处理输入，`forward_generator` 只返回 loss 和 `student_sample`，由外层 `forward` 返回准备好的 `inputs` 供 critic 复用，不重复预处理；生成样本 `student_sample` 单独返回并作为参数传给 critic，不写入 `inputs_shared`。`launch_dmd2_training_task` 按顺序调用两阶段的 `model(...)` 并反向传播，负责梯度累积、参数更新和保存。critic 的损失合并为一次反传；每个微批次在更新学生时反传两次，否则一次。

模型创建、LoRA、`UnifiedDataset`、`Accelerator` 和 `ModelLogger` 使用已有框架实现。

公共 trainer 通过 `lora_base_model` 选择主干，LoRA 注入、参数收集、GAN hook 和权重保存加载均使用该名称。`feature_module_of_dit_for_gan` 指定相对于所选主干的模块路径。

`prepare_batch` 必须由各模型的 trainer 实现，基类只声明接口并抛出 `NotImplementedError`。返回值为 `(inputs_shared, inputs_posi, inputs_nega)`，其中共享输入需提供 `noise`，开启 GAN 时还需提供真实样本的 `input_latents`。输入图像或视频等原始字段由子类处理，公共 trainer 不指定。

MOSS 的实现按 SFT 的路径流转：`get_pipeline_inputs(data)` → `transfer_data_to_device` → `pipe.unit_runner`。无 GAN 时将 `input_image` 设为 `None`，图片仍提供宽高，pipeline 不编码真实图像。

checkpoint 配置沿用 SFT 的 `get_pipeline_inputs`，保存在 `inputs_shared` 中，不在 rollout 和 loss 各层重复传参。公共 `predict` 收集 `pipe.in_iteration_models`，复制共享字典并替换当前 `latents`，checkpoint 配置原样传给 `model_fn`，通过原有 `pipe.model_fn` 与 `pipe.cfg_guided_model_fn` 完成预测。teacher 使用配置的 CFG，student/fake 使用 CFG=1；正负条件分别来自当前样本的输入字典，不另建条件格式，也不全局缓存负条件。MOSS 的 `inputs_nega["skip_layers"]=(22,)` 直接沿用 SFT，文本长度保持 4096，GAN 的 VAE 保持 float32。

本示例展示 MOSS 对公共 trainer 的适配。接入已有 SFT 模型时，子类实现 `prepare_batch`，复用已有输入处理和 pipeline units；`predict` 继续使用公共实现。启用 GAN 时需提供 `gan_feature_channels` 和 `reshape_gan_features`，把模型 hook 输出转为 BCHW。噪声和 latent 保留 pipeline 的原生布局，main 与数据集构造仍留在模型示例中。

## 运行示例

在仓库根目录运行公开示例：

```bash
PYTHONPATH="./" bash examples/moss_image/model_training/special/dmd2_trainer/train_MOSS-Image-SFT-8steps_lora.sh
```

脚本依次提供运行说明、数据下载、时间序列计算和训练命令。下载 `DiffSynth-Studio/diffsynth_example_dataset` 中的 `moss_image/MOSS-Image-SFT/*`，保存到 `data/diffsynth_example_dataset/moss_image/MOSS-Image-SFT/`；训练读取该目录下的 `metadata.csv`。每条记录提供 `prompt` 和 `image`，图像路径相对于上述目录，不要求额外宽高字段。

脚本通过 `accelerate launch` 启动，进程数由 Accelerate 配置决定；没有固定为八卡。每个进程 micro-batch 为 1，梯度累积 2，有效 batch 为 `进程数 × 2`。student/fake 均使用 LoRA、固定学习率和单优化器，图像预处理沿用框架实现。

| 配置 | 默认值 |
|---|---|
| 学生轨迹 | 普通 Euler，8 步，显式指定 sigmas 与 timesteps |
| Student / fake LoRA rank | 均为 128，alpha 等于 rank |
| 每进程 batch / 累积 | 1 / 2，有效 batch 为进程数 × 2 |
| 学习率 | `5e-6`，单个优化器统一使用，固定不变 |
| fake:G 更新比例 | 5:1 |
| teacher CFG | 3 |
| GAN 特征模块 | `layers.21` |
| 最大像素 | 1,048,576（1024×1024） |
| 数据 repeat / workers / epochs | 50 / 8 / 5 |
| weight decay | 框架默认值 0.01，脚本不传入 |

Student 和 fake LoRA 的 alpha 始终等于各自的 rank。LoRA 目标使用逗号分隔的模块后缀 `proj.q,proj.k,proj.v,proj.o,proj.ffn_up_gate,proj.ffn_down`，对应每个 DiT block 的六个投影层；保留 `proj.` 前缀，避免匹配文本模块中的同名层。

示例最大像素为 1,048,576，并在启动前通过 `examples/dev_tools/compute_dmd2_sigmas.py` 打印原生 MOSS 时间序列。工具参数与 OPD 的计算脚本一致：`--scheduler_name "MOSS-Image" --num_inference_steps 8 --extra_kwargs '{"shift": 5.0}'`。调整步数或 shift 后，将计算结果填入脚本的学生参数。

学生时间序列由脚本传入：

```text
sigmas:    1.0, 0.9679999947547913, 0.9259999990463257, 0.8709999918937683, 0.7910000085830688, 0.6710000038146973, 0.46300002932548523, 0.024000000208616257
timesteps: 1000.0, 968.0, 926.0, 871.0, 791.0, 671.0, 463.0000305175781, 24.0
```

分别对应 `--dmd2_student_sigmas` 和 `--dmd2_student_timesteps`；最后一步推进至 sigma 0。teacher 在损失采样的噪声时间计算预测，不运行完整去噪轨迹，因此不提供 teacher 轨迹参数。

需要调整训练配置时，直接修改脚本中的对应参数。示例输出写入固定目录 `./models/train/MOSS-Image-SFT_dmd2_lora_8steps`，可通过 `--output_path` 调整。直接运行 Python 时，请显式传入 `--output_path` 和交替训练所需的 `--find_unused_parameters`。

GAN 默认开启，在训练命令中加入 `--disable_gan` 即可关闭。关闭时保留 fake-score 训练，读取图片只用于确定训练尺寸，不运行 VAE 编码，也不把图片像素用于训练损失。脚本通过 `--feature_module_of_dit_for_gan layers.21` 指定 fake 的特征 hook，转为 BCHW；分类头自适应池化到 32×32。

两种模式均使用 `UnifiedDataset.default_image_operator` 读取和处理图片，统一遵循 `--max_pixels`、`--height`、`--width` 与默认动态尺寸的 16 倍数对齐规则。无 GAN 时使用处理后图片的宽高创建随机 latent。

`--generator_gan_loss_weight` 默认 0.005，控制生成器 GAN loss；`--discriminator_gan_loss_weight` 默认 0.01，控制判别端真实与生成样本的 GAN loss。判别端梯度更新 fake LoRA 和分类头，DiT 主干冻结。

开启 gradient checkpointing 时，先通过 fake 模型计算 GAN 对生成样本的梯度，再切回 student LoRA 反向传播，保证两段重算各自使用正确的模型状态。

模型和优化器使用训练设备，直接通过 `Accelerator.prepare` 和 `accumulate` 训练。每个微批次调用 `optimizer.step()`、`scheduler.step()` 和 `zero_grad()`，由 Accelerate 在累积完成时执行实际更新；fake 每个窗口更新，student 每五个窗口更新一次，其余窗口保持 student 梯度为 `None`，AdamW 跳过这些参数。训练轮数由框架的 `--num_epochs` 控制；student 与 fake/分类头仅在累积完成时分别裁剪梯度，阈值均为 10。

## 保存与继续训练

`--save_steps 100` 按 **optimizer 实际更新次数**计数，每完成一个梯度累积窗口计 1，无论该窗口只更新 fake 还是同时更新 G 和 fake；每 100 次更新保存一次。训练使用主进程的 `tqdm` 进度条，按数据批次推进，附加信息显示本次训练累计的 G 更新次数 `generator_updates`，不记录 loss。使用框架 `ModelLogger` 输出 `step-100.safetensors` 等文件，只保存学生 LoRA，可直接通过 `pipe.load_lora` 使用普通 Euler 推理；推理使用与训练相同的学生时间序列。

通过 `--lora_checkpoint /path/to/step-100.safetensors` 热启动学生。这里不提供完整训练状态恢复，不恢复 fake、GAN 分类头、优化器或数据游标。加载学生 LoRA 后，普通 Euler 推理使用 `num_inference_steps=8, sigma_shift=5, cfg_scale=1`。
