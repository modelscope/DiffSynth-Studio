# `diffsynth.core.gradient`: 梯度检查点及其 Offload

`diffsynth.core.gradient` 中提供了封装好的梯度检查点及其 Offload 版本，用于模型训练。

## 梯度检查点

梯度检查点是用于减少训练时显存占用的技术。我们提供一个例子来帮助你理解这一技术，以下是一个简单的模型结构

```python
import torch

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.Sigmoid()
    
    def forward(self, x):
        return self.activation(x)

model = ToyModel()
x = torch.randn((2, 3))
y = model(x)
```

在这个模型结构中，输入的参数 $x$ 经过 Sigmoid 激活函数得到输出值 $y=\frac{1}{1+e^{-x}}$。

在训练过程中，假定我们的损失函数值为 $\mathcal L$，在梯度反响传播时，我们得到 $\frac{\partial \mathcal L}{\partial y}$，此时我们需计算 $\frac{\partial \mathcal L}{\partial x}$，不难发现 $\frac{\partial y}{\partial x}=y(1-y)$，进而有 $\frac{\partial \mathcal L}{\partial x}=\frac{\partial \mathcal L}{\partial y}\frac{\partial y}{\partial x}=\frac{\partial \mathcal L}{\partial y}y(1-y)$。如果在模型前向传播时保存 $y$ 的数值，并在梯度反向传播时直接计算 $y(1-y)$，这将避免复杂的 exp 计算，加快计算速度，但这会导致我们需要额外的显存来存储中间变量 $y$。

不启用梯度检查点时，训练框架会默认存储所有辅助梯度计算的中间变量，从而达到最佳的计算速度。开启梯度检查点时，中间变量则不会存储，但输入参数 $x$ 仍会存储，减少显存占用，在梯度反向传播时需重新计算这些变量，减慢计算速度。

## 启用梯度检查点及其 Offload

`diffsynth.core.gradient` 中的 `gradient_checkpoint_forward` 实现了梯度检查点及其 Offload，可参考以下代码调用：

```python
import torch
from diffsynth.core.gradient import gradient_checkpoint_forward

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.Sigmoid()
    
    def forward(self, x):
        return self.activation(x)

model = ToyModel()
x = torch.randn((2, 3))
y = gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing=True,
    use_gradient_checkpointing_offload=False,
    x=x,
)
```

* 当 `use_gradient_checkpointing=False` 且 `use_gradient_checkpointing_offload=False` 时，计算过程与原始计算完全相同，不影响模型的推理和训练，你可以直接将其集成到代码中。
* 当 `use_gradient_checkpointing=True` 且 `use_gradient_checkpointing_offload=False` 时，启用梯度检查点。
* 当 `use_gradient_checkpointing_offload=True` 时，启用梯度检查点，所有梯度检查点的输入参数存储在内存中，进一步降低显存占用和减慢计算速度。

## 最佳实践

> Q: 应当在何处启用梯度检查点？
> 
> A: 对整个模型启用梯度检查点时，计算效率和显存占用并不是最优的，我们需要设置细粒度的梯度检查点，但同时不希望为框架增加过多繁杂的代码。因此我们建议在 `Pipeline` 的 `model_fn` 中实现，例如 `diffsynth/pipelines/qwen_image.py` 中的 `model_fn_qwen_image`，在 Block 层级启用梯度检查点，不需要修改模型结构的任何代码。

> Q: 什么情况下需要启用梯度检查点？
> 
> A: 随着模型参数量越来越大，梯度检查点已成为必要的训练技术，梯度检查点通常是需要启用的。梯度检查点的 Offload 则仅需在激活值占用显存过大的模型（例如视频生成模型）中启用。
