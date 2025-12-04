# 差分 LoRA 训练

差分 LoRA 训练是一种特殊的 LoRA 训练方式，旨在让模型学习图像之间的差异。

## 训练方案

我们未能找到差分 LoRA 训练最早由谁提出，这一技术已经在开源社区中流传甚久。

假设我们有两张内容相似的图像：图 1 和图 2。例如两张图中分别有一辆车，但图 1 中画面细节更少，图 2 中画面细节更多。在差分 LoRA 训练中，我们进行两步训练：

* 以图 1 为训练数据，以[标准监督训练](/docs/zh/Training/Supervised_Fine_Tuning.md)的方式，训练 LoRA 1
* 以图 2 为训练数据，将 LoRA 1 融入基础模型后，以[标准监督训练](/docs/zh/Training/Supervised_Fine_Tuning.md)的方式，训练 LoRA 2

在第一步训练中，由于训练数据仅有一张图，LoRA 模型很容易过拟合，因此训练完成后，LoRA 1 会让模型毫不犹豫地生成图 1，无论随机种子是什么。在第二步训练中，LoRA 模型再次过拟合，因此训练完成后，在 LoRA 1 和 LoRA 2 的共同作用下，模型会毫不犹豫地生成图 2。简言之：

* LoRA 1 = 生成图 1
* LoRA 1 + LoRA 2 = 生成图 2

此时丢弃 LoRA 1，只使用 LoRA 2，模型将会理解图 1 和图 2 的差异，使生成的内容倾向于“更不像图1，更像图 2”。

单一训练数据可以保证模型能够过拟合到训练数据上，但稳定性不足。为了提高稳定性，我们可以用多个图像对（image pairs）进行训练，并将训练出的 LoRA 2 进行平均，得到效果更稳定的 LoRA。

用这一训练方案，可以训练出一些功能奇特的 LoRA 模型。例如，使用丑陋的和漂亮的图像对，训练提升图像美感的 LoRA；使用细节少的和细节丰富的图像对，训练增加图像细节的 LoRA。

## 模型效果

我们用差分 LoRA 训练技术训练了几个美学提升 LoRA，可前往对应的模型页面查看生成效果。

* [DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1)
* [DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1](https://modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1)

## 在训练框架中使用差分 LoRA 训练

第一步的训练与普通 LoRA 训练没有任何差异，在第二步的训练命令中，通过 `--preset_lora_path` 参数填入第一步的 LoRA 模型文件路径，并将 `--preset_lora_model` 设置为与 `lora_base_model` 相同的参数，即可将 LoRA 1 加载到基础模型中。

## 框架设计思路

在训练框架中，`--preset_lora_path` 指向的模型在 `DiffusionTrainingModule` 的 `switch_pipe_to_training_mode` 中完成加载。
