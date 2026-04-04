# `diffsynth.core.gradient`: Gradient Checkpointing and Offload

`diffsynth.core.gradient` provides encapsulated gradient checkpointing and its Offload version for model training.

## Gradient Checkpointing

Gradient checkpointing is a technique used to reduce memory usage during training. We provide an example to help you understand this technique. Here is a simple model structure:

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

In this model structure, the input parameter $x$ passes through the Sigmoid activation function to obtain the output value $y=\frac{1}{1+e^{-x}}$.

During the training process, assuming our loss function value is $\mathcal L$, when backpropagating gradients, we obtain $\frac{\partial \mathcal L}{\partial y}$. At this point, we need to calculate $\frac{\partial \mathcal L}{\partial x}$. It's not difficult to find that $\frac{\partial y}{\partial x}=y(1-y)$, and thus $\frac{\partial \mathcal L}{\partial x}=\frac{\partial \mathcal L}{\partial y}\frac{\partial y}{\partial x}=\frac{\partial \mathcal L}{\partial y}y(1-y)$. If we save the value of $y$ during the model's forward propagation and directly compute $y(1-y)$ during gradient backpropagation, this will avoid complex exp computations, speeding up the calculation. However, this requires additional memory to store the intermediate variable $y$.

When gradient checkpointing is not enabled, the training framework will default to storing all intermediate variables that assist gradient computation, thereby achieving optimal computational speed. When gradient checkpointing is enabled, intermediate variables are not stored, but the input parameter $x$ is still stored, reducing memory usage. During gradient backpropagation, these variables need to be recomputed, slowing down the calculation.

## Enabling Gradient Checkpointing and Its Offload

`gradient_checkpoint_forward` in `diffsynth.core.gradient` implements gradient checkpointing and its Offload. Refer to the following code for calling:

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

* When `use_gradient_checkpointing=False` and `use_gradient_checkpointing_offload=False`, the computation process is exactly the same as the original computation, not affecting the model's inference and training. You can directly integrate it into your code.
* When `use_gradient_checkpointing=True` and `use_gradient_checkpointing_offload=False`, gradient checkpointing is enabled.
* When `use_gradient_checkpointing_offload=True`, gradient checkpointing is enabled, and all gradient checkpoint input parameters are stored in memory, further reducing memory usage and slowing down computation.

## Best Practices

> Q: Where should gradient checkpointing be enabled?
> 
> A: When enabling gradient checkpointing for the entire model, computational efficiency and memory usage are not optimal. We need to set fine-grained gradient checkpoints, but we don't want to add too much complicated code to the framework. Therefore, we recommend implementing it in the `model_fn` of `Pipeline`, for example, `model_fn_qwen_image` in `diffsynth/pipelines/qwen_image.py`, enabling gradient checkpointing at the Block level without modifying any code in the model structure.

> Q: When should gradient checkpointing be enabled?
> 
> A: As model parameters become increasingly large, gradient checkpointing has become a necessary training technique. Gradient checkpointing usually needs to be enabled. Gradient checkpointing Offload should only be enabled in models where activation values occupy excessive memory (such as video generation models).