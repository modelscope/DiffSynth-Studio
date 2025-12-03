# Model Training

This document introduces how to use `DiffSynth-Studio` for model training.

## Script Parameters

Training scripts typically include the following parameters:

* Dataset base configuration
    * `--dataset_base_path`: Root directory of the dataset.
    * `--dataset_metadata_path`: Metadata file path of the dataset.
    * `--dataset_repeat`: Number of times the dataset is repeated in each epoch.
    * `--dataset_num_workers`: Number of processes for each Dataloader.
    * `--data_file_keys`: Field names that need to be loaded from metadata, usually image or video file paths, separated by `,`.
* Model loading configuration
    * `--model_paths`: Paths of models to be loaded. JSON format.
    * `--model_id_with_origin_paths`: Model IDs with original paths, for example `"Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors"`. Separated by commas.
    * `--extra_inputs`: Extra input parameters required by the model Pipeline, for example, training image editing model Qwen-Image-Edit requires extra parameter `edit_image`, separated by `,`.
    * `--fp8_models`: Models loaded in FP8 format, consistent with the format of `--model_paths` or `--model_id_with_origin_paths`. Currently only supports models whose parameters are not updated by gradients (no gradient backpropagation, or gradients only update their LoRA).
* Training base configuration
    * `--learning_rate`: Learning rate.
    * `--num_epochs`: Number of epochs.
    * `--trainable_models`: Trainable models, for example `dit`, `vae`, `text_encoder`.
    * `--find_unused_parameters`: Whether there are unused parameters in DDP training. Some models contain redundant parameters that do not participate in gradient calculation, and this setting needs to be enabled to avoid errors in multi-GPU training.
    * `--weight_decay`: Weight decay size. See [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) for details.
    * `--task`: Training task, default is `sft`. Some models support more training modes. Please refer to the documentation for each specific model.
* Output configuration
    * `--output_path`: Model save path.
    * `--remove_prefix_in_ckpt`: Remove prefixes in the state dict of model files.
    * `--save_steps`: Interval of training steps for saving models. If this parameter is left blank, the model will be saved once per epoch.
* LoRA configuration
    * `--lora_base_model`: Which model LoRA is added to.
    * `--lora_target_modules`: Which layers LoRA is added to.
    * `--lora_rank`: Rank of LoRA.
    * `--lora_checkpoint`: Path of LoRA checkpoint. If this path is provided, LoRA will be loaded from this checkpoint.
    * `--preset_lora_path`: Preset LoRA checkpoint path. If this path is provided, this LoRA will be loaded in the form of being merged into the base model. This parameter is used for LoRA differential training.
    * `--preset_lora_model`: Model that preset LoRA is merged into, for example `dit`.
* Gradient configuration
    * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
    * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to memory.
    * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
* Image dimension configuration (applicable to image generation models and video generation models)
    * `--height`: Height of images or videos. Leave `height` and `width` blank to enable dynamic resolution.
    * `--width`: Width of images or videos. Leave `height` and `width` blank to enable dynamic resolution.
    * `--max_pixels`: Maximum pixel area of images or video frames. When dynamic resolution is enabled, images with resolution larger than this value will be scaled down, and images with resolution smaller than this value will remain unchanged.

Some models' training scripts also contain additional parameters. See the documentation for each model for details.

## Preparing Datasets

`DiffSynth-Studio` adopts a universal dataset format. The dataset contains a series of data files (images, videos, etc.) and annotated metadata files. We recommend organizing dataset files as follows:

```
data/example_image_dataset/
├── metadata.csv
├── image_1.jpg
└── image_2.jpg
```

Where `image_1.jpg`, `image_2.jpg` are training image data, and `metadata.csv` is the metadata list, for example:

```
image,prompt
image_1.jpg,"a dog"
image_2.jpg,"a cat"
```

We have built sample datasets for your testing. To understand how the universal dataset architecture is implemented, please refer to [`diffsynth.core.data`](/docs/en/API_Reference/core/data.md).

<details>

<summary>Sample Image Dataset</summary>

> ```shell
> modelscope download --dataset DiffSynth-Studio/example_image_dataset --local_dir ./data/example_image_dataset
> ```
> 
> Applicable to training of image generation models such as Qwen-Image and FLUX.

</details>

<details>

<summary>Sample Video Dataset</summary>

> ```shell
> modelscope download --dataset DiffSynth-Studio/example_video_dataset --local_dir ./data/example_video_dataset
> ```
> 
> Applicable to training of video generation models such as Wan.

</details>

## Loading Models

Similar to [model loading during inference](/docs/en/Pipeline_Usage/Model_Inference.md#loading-models), we support multiple ways to configure model paths, and the two methods can be mixed.

<details>

<summary>Download and load models from remote sources</summary>

> If we load models during inference through the following settings:
> 
> ```python
> model_configs=[
>     ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
>     ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
>     ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
> ]
> ```
> 
> Then during training, fill in the following parameters to load the corresponding models:
> 
> ```shell
> --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"
> ```
> 
> Model files are downloaded to the `./models` path by default, which can be modified through [environment variable DIFFSYNTH_MODEL_BASE_PATH](/docs/en/Pipeline_Usage/Environment_Variables.md#diffsynth_model_base_path).
> 
> By default, even after models have been downloaded, the program will still query remotely for missing files. To completely disable remote requests, set [environment variable DIFFSYNTH_SKIP_DOWNLOAD](/docs/en/Pipeline_Usage/Environment_Variables.md#diffsynth_skip_download) to `True`.

</details>

<details>

<details>

<summary>Load models from local file paths</summary>

> If loading models from local files during inference, for example:
> 
> ```python
> model_configs=[
>     ModelConfig([
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
>     ]),
>     ModelConfig([
>         "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
>     ]),
>     ModelConfig("models/Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors")
> ]
> ```
> 
> Then during training, set to:
> 
> ```shell
> --model_paths '[
>     [
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
>     ],
>     [
>         "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
>     ],
>     "models/Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors"
> ]' \
> ```
> 
> Note that `--model_paths` is in JSON format, and extra `,` cannot appear in it, otherwise it cannot be parsed normally.

</details>

## Setting Trainable Modules

The training framework supports training of any model. Taking Qwen-Image as an example, to fully train the DiT model, set to:

```shell
--trainable_models "dit"
```

To train LoRA of the DiT model, set to:

```shell
--lora_base_model dit --lora_target_modules "to_q,to_k,to_v" --lora_rank 32
```

We hope to leave enough room for technical exploration, so the framework supports training any number of modules simultaneously. For example, to train the text encoder, controlnet, and LoRA of the DiT simultaneously:

```shell
--trainable_models "text_encoder,controlnet" --lora_base_model dit --lora_target_modules "to_q,to_k,to_v" --lora_rank 32
```

Additionally, since the training script loads multiple modules (text encoder, dit, vae, etc.), prefixes need to be removed when saving model files. For example, when fully training the DiT part or training the LoRA model of the DiT part, please set `--remove_prefix_in_ckpt pipe.dit.`. If multiple modules are trained simultaneously, developers need to write code to split the state dict in the model file after training is completed.

## Starting the Training Program

The training framework is built on [`accelerate`](https://huggingface.co/docs/accelerate/index). Training commands are written in the following format:

```shell
accelerate launch xxx/train.py \
  --xxx yyy \
  --xxxx yyyy
```

We have written preset training scripts for each model. See the documentation for each model for details.

By default, `accelerate` will train according to the configuration in `~/.cache/huggingface/accelerate/default_config.yaml`. Use `accelerate config` to configure interactively in the terminal, including multi-GPU training, [`DeepSpeed`](https://www.deepspeed.ai/), etc.

We provide recommended `accelerate` configuration files for some models, which can be set through `--config_file`. For example, full training of the Qwen-Image model:

```shell
accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters
```

## Training Considerations

* In addition to the `csv` format, dataset metadata also supports `json` and `jsonl` formats. For how to choose the best metadata format, please refer to [/docs/en/API_Reference/core/data.md#metadata](/docs/en/API_Reference/core/data.md#metadata)
* Training effectiveness is usually strongly correlated with training steps and weakly correlated with epoch count. Therefore, we recommend using the `--save_steps` parameter to save model files at training step intervals.
* When data volume * `dataset_repeat` exceeds $10^9$, we observed that the dataset speed becomes significantly slower, which seems to be a `PyTorch` bug. We are not sure if newer versions of `PyTorch` have fixed this issue.
* For learning rate `--learning_rate`, it is recommended to set to `1e-4` in LoRA training and `1e-5` in full training.
* The training framework does not support batch size > 1. The reasons are complex. See [Q&A: Why doesn't the training framework support batch size > 1?](/docs/en/QA.md#why-doesnt-the-training-framework-support-batch-size--1)
* Some models contain redundant parameters. For example, the text encoding part of the last layer of Qwen-Image's DiT part. When training these models, `--find_unused_parameters` needs to be set to avoid errors in multi-GPU training. For compatibility with community models, we do not intend to remove these redundant parameters.
* The loss function value of Diffusion models has little relationship with actual effects. Therefore, we do not record loss function values during training. We recommend setting `--num_epochs` to a sufficiently large value, testing while training, and manually closing the training program after the effect converges.
* `--use_gradient_checkpointing` is usually enabled unless GPU VRAM is sufficient; `--use_gradient_checkpointing_offload` is enabled as needed. See [`diffsynth.core.gradient`](/docs/en/API_Reference/core/gradient.md) for details.