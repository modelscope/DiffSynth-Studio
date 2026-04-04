# Standard Supervised Training

After understanding the [Basic Principles of Diffusion Models](/docs/en/Training/Understanding_Diffusion_models.md), this document introduces how the framework implements Diffusion model training. This document explains the framework's principles to help developers write new training code. If you want to use our provided default training functions, please refer to [Model Training](/docs/en/Pipeline_Usage/Model_Training.md).

Recalling the model training pseudocode from earlier, when we actually write code, the situation becomes extremely complex. Some models require additional guidance conditions and preprocessing, such as ControlNet; some models require cross-computation with the denoising model, such as VACE; some models require Gradient Checkpointing due to excessive VRAM demands, such as Qwen-Image's DiT.

To achieve strict consistency between inference and training, we abstractly encapsulate components like `Pipeline`, reusing inference code extensively during training. Please refer to [Integrating Pipeline](/docs/en/Developer_Guide/Building_a_Pipeline.md) to understand the design of `Pipeline` components. Next, we'll introduce how the training framework utilizes `Pipeline` components to build training algorithms.

## Framework Design Concept

The training module is encapsulated on top of the `Pipeline`, inheriting `DiffusionTrainingModule` from `diffsynth.diffusion.training_module`. We need to provide the necessary `__init__` and `forward` methods for the training module. Taking Qwen-Image's LoRA training as an example, we provide a simple script containing only basic training functions in `examples/qwen_image/model_training/special/simple/train.py` to help developers understand the design concept of the training module.

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

In `__init__`, model initialization is required. First load the model, then switch it to training mode.

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

The logic for loading models is basically consistent with inference, supporting loading models from remote and local paths. See [Model Inference](/docs/en/Pipeline_Usage/Model_Inference.md) for details, but please note not to enable [VRAM Management](/docs/en/Pipeline_Usage/VRAM_management.md).

`switch_pipe_to_training_mode` can switch the model to training mode. See `switch_pipe_to_training_mode` for details.

### `forward`

In `forward`, the loss function value needs to be calculated. First perform preprocessing, then compute the loss function through the `Pipeline`'s [`model_fn`](/docs/en/Developer_Guide/Building_a_Pipeline.md#model_fn).

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

The preprocessing process is consistent with the inference phase. Developers only need to assume they are using the `Pipeline` for inference and fill in the input parameters.

The loss function calculation reuses `FlowMatchSFTLoss` from `diffsynth.diffusion.loss`.

### Starting Training

The training framework requires other modules, including:

* accelerator: Training launcher provided by `accelerate`, see [`accelerate`](https://huggingface.co/docs/accelerate/index) for details
* dataset: Generic dataset, see [`diffsynth.core.data`](/docs/en/API_Reference/core/data.md) for details
* model_logger: Model logger, see `diffsynth.diffusion.logger` for details

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

Assembling all the above code results in `examples/qwen_image/model_training/special/simple/train.py`. Use the following command to start training:

```
accelerate launch examples/qwen_image/model_training/special/simple/train.py
```