# 标准监督训练

在理解 [Diffusion 模型基本原理](/docs/zh/Training/Understanding_Diffusion_models.md)之后，本文档介绍框架如何实现 Diffusion 模型的训练。本文档介绍框架的原理，帮助开发者编写新的训练代码，如需使用我们提供的默认训练功能，请参考[模型训练](/docs/zh/Pipeline_Usage/Model_Training.md)。

回顾前文中的模型训练伪代码，当我们实际编写代码时，情况会变得极为复杂。部分模型需要输入额外的引导条件并进行预处理，例如 ControlNet；部分模型需要与去噪模型进行交叉式的计算，例如 VACE；部分模型因显存需求过大，需要开启 Gradient Checkpointing，例如 Qwen-Image 的 DiT。

为了实现严格的推理和训练一致性，我们对 `Pipeline` 等组件进行了抽象封装，在训练过程中大量复用推理代码。请参考[接入 Pipeline](/docs/zh/Developer_Guide/Building_a_Pipeline.md) 了解 `Pipeline` 组件的设计。接下来我们介绍训练框架如何利用 `Pipeline` 组件构建训练算法。

## 框架设计思路

训练模块在 `Pipeline` 上层进行封装，继承 `diffsynth.diffusion.training_module` 中的 `DiffusionTrainingModule`，我们需为训练模块提供必要的 `__init__` 和 `forward` 方法。我们以 Qwen-Image 的 LoRA 训练为例，在 `examples/qwen_image/model_training/special/simple/train.py` 中提供了仅包含基础训练功能的简易脚本，帮助开发者理解训练模块的设计思路。

```python
class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(self, device):
        # Initialize models here.
        pass

    def forward(self, data):
        # Compute loss here.
        return loss
```

### `__init__`

在 `__init__` 中需进行模型的初始化，先加载模型，然后将其切换到训练模式。

```python
    def __init__(self, device):
        super().__init__()
        # Load the pipeline
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        )
        # Switch to training mode
        self.switch_pipe_to_training_mode(
            self.pipe,
            lora_base_model="dit",
            lora_target_modules="to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj",
            lora_rank=32,
        )
```

加载模型的逻辑与推理时基本一致，支持从远程和本地路径加载模型，详见[模型推理](/docs/zh/Pipeline_Usage/Model_Inference.md)，但请注意不要启用[显存管理](/docs/zh/Pipeline_Usage/VRAM_management.md)。

`switch_pipe_to_training_mode` 可以将模型切换到训练模式，详见 `switch_pipe_to_training_mode`。

### `forward`

在 `forward` 中需计算损失函数值，先进行前处理，然后经过 `Pipeline` 的 [`model_fn`](/docs/zh/Developer_Guide/Building_a_Pipeline.md#model_fn) 计算损失函数。

```python
    def forward(self, data):
        # Preprocess
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": True,
            "use_gradient_checkpointing_offload": False,
        }
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        # Loss
        loss = FlowMatchSFTLoss(self.pipe, **inputs_shared, **inputs_posi)
        return loss
```

前处理过程与推理阶段一致，开发者只需假定在使用 `Pipeline` 进行推理，将输入参数填入即可。

损失函数的计算沿用 `diffsynth.diffusion.loss` 中的 `FlowMatchSFTLoss`。

### 开始训练

训练框架还需其他模块，包括：

* accelerator: `accelerate` 提供的训练启动器，详见 [`accelerate`](https://huggingface.co/docs/accelerate/index)
* dataset: 通用数据集，详见 [`diffsynth.core.data`](/docs/zh/API_Reference/core/data.md)
* model_logger: 模型记录器，详见 `diffsynth.diffusion.logger`

```python
if __name__ == "__main__":
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    dataset = UnifiedDataset(
        base_path="data/example_image_dataset",
        metadata_path="data/example_image_dataset/metadata.csv",
        repeat=50,
        data_file_keys="image",
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path="data/example_image_dataset",
            height=512,
            width=512,
            height_division_factor=16,
            width_division_factor=16,
        )
    )
    model = QwenImageTrainingModule(accelerator.device)
    model_logger = ModelLogger(
        output_path="models/toy_model",
        remove_prefix_in_ckpt="pipe.dit.",
    )
    launch_training_task(
        accelerator, dataset, model, model_logger,
        learning_rate=1e-5, num_epochs=1,
    )
```

将以上所有代码组装，得到 `examples/qwen_image/model_training/special/simple/train.py`。使用以下命令即可启动训练：

```
accelerate launch examples/qwen_image/model_training/special/simple/train.py
```
