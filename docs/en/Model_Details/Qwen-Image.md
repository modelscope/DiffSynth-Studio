# Qwen-Image

![Image](https://github.com/user-attachments/assets/738078d8-8749-4a53-a046-571861541924)

Qwen-Image is an image generation model trained and open-sourced by the Tongyi Lab Qwen Team of Alibaba.

## Installation

Before using this project for model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information about installation, please refer to [Install Dependencies](/docs/en/Pipeline_Usage/Setup.md).

## Quick Start

Run the following code to quickly load the [Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image) model and perform inference. VRAM management is enabled, and the framework will automatically control model parameter loading based on remaining VRAM. Minimum 8GB VRAM is required to run.

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

## Model Overview

<details>

<summary>Model Lineage</summary>

```mermaid
graph LR;
    Qwen/Qwen-Image-->Qwen/Qwen-Image-Edit;
    Qwen/Qwen-Image-Edit-->Qwen/Qwen-Image-Edit-2509;
    Qwen/Qwen-Image-->EliGen-Series;
    EliGen-Series-->DiffSynth-Studio/Qwen-Image-EliGen;
    DiffSynth-Studio/Qwen-Image-EliGen-->DiffSynth-Studio/Qwen-Image-EliGen-V2;
    EliGen-Series-->DiffSynth-Studio/Qwen-Image-EliGen-Poster;
    Qwen/Qwen-Image-->Distill-Series;
    Distill-Series-->DiffSynth-Studio/Qwen-Image-Distill-Full;
    Distill-Series-->DiffSynth-Studio/Qwen-Image-Distill-LoRA;
    Qwen/Qwen-Image-->ControlNet-Series;
    ControlNet-Series-->Blockwise-ControlNet-Series;
    Blockwise-ControlNet-Series-->DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny;
    Blockwise-ControlNet-Series-->DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth;
    Blockwise-ControlNet-Series-->DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint;
    ControlNet-Series-->DiffSynth-Studio/Qwen-Image-In-Context-Control-Union;
    Qwen/Qwen-Image-->DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix;
```

</details>

| Model ID | Inference | Low VRAM Inference | Full Training | Validation After Full Training | LoRA Training | Validation After LoRA Training |
| - | - | - | - | - | - | - |
| [Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image) | [code](/examples/qwen_image/model_inference/Qwen-Image.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image.py) |
|[Qwen/Qwen-Image-2512](https://www.modelscope.cn/models/Qwen/Qwen-Image-2512)|[code](/examples/qwen_image/model_inference/Qwen-Image-2512.py)|[code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-2512.py)|[code](/examples/qwen_image/model_training/full/Qwen-Image-2512.sh)|[code](/examples/qwen_image/model_training/validate_full/Qwen-Image-2512.py)|[code](/examples/qwen_image/model_training/lora/Qwen-Image-2512.sh)|[code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-2512.py)|
| [Qwen/Qwen-Image-Edit](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Edit.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Edit.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Edit.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit.py) |
| [Qwen/Qwen-Image-Edit-2509](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2509) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit-2509.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2509.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Edit-2509.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Edit-2509.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2509.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2509.py) |
|[Qwen/Qwen-Image-Edit-2511](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2511)|[code](/examples/qwen_image/model_inference/Qwen-Image-Edit-2511.py)|[code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2511.py)|[code](/examples/qwen_image/model_training/full/Qwen-Image-Edit-2511.sh)|[code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Edit-2511.py)|[code](/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh)|[code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2511.py)|
|[lightx2v/Qwen-Image-Edit-2511-Lightning](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning)|[code](/examples/qwen_image/model_inference/Qwen-Image-Edit-2511-Lightning.py)|[code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2511-Lightning.py)|-|-|-|-|
|[Qwen/Qwen-Image-Layered](https://www.modelscope.cn/models/Qwen/Qwen-Image-Layered)|[code](/examples/qwen_image/model_inference/Qwen-Image-Layered.py)|[code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Layered.py)|[code](/examples/qwen_image/model_training/full/Qwen-Image-Layered.sh)|[code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Layered.py)|[code](/examples/qwen_image/model_training/lora/Qwen-Image-Layered.sh)|[code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Layered.py)|
|[DiffSynth-Studio/Qwen-Image-Layered-Control](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control)|[code](/examples/qwen_image/model_inference/Qwen-Image-Layered-Control.py)|[code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Layered-Control.py)|[code](/examples/qwen_image/model_training/full/Qwen-Image-Layered-Control.sh)|[code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Layered-Control.py)|[code](/examples/qwen_image/model_training/lora/Qwen-Image-Layered-Control.sh)|[code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Layered-Control.py)|
| [DiffSynth-Studio/Qwen-Image-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen) | [code](/examples/qwen_image/model_inference/Qwen-Image-EliGen.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-EliGen.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-EliGen.py) |
| [DiffSynth-Studio/Qwen-Image-EliGen-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-V2) | [code](/examples/qwen_image/model_inference/Qwen-Image-EliGen-V2.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen-V2.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-EliGen.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-EliGen.py) |
| [DiffSynth-Studio/Qwen-Image-EliGen-Poster](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-Poster) | [code](/examples/qwen_image/model_inference/Qwen-Image-EliGen-Poster.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen-Poster.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-EliGen-Poster.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-EliGen-Poster.py) |
| [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full) | [code](/examples/qwen_image/model_inference/Qwen-Image-Distill-Full.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Distill-Full.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Distill-Full.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Distill-Full.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Distill-Full.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Distill-Full.py) |
| [DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA) | [code](/examples/qwen_image/model_inference/Qwen-Image-Distill-LoRA.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Distill-LoRA.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Distill-LoRA.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Distill-LoRA.py) |
| [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny) | [code](/examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Canny.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Canny.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Blockwise-ControlNet-Canny.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Canny.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Blockwise-ControlNet-Canny.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Canny.py) |
| [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth) | [code](/examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Depth.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Depth.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Blockwise-ControlNet-Depth.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Depth.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Blockwise-ControlNet-Depth.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Depth.py) |
| [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint) | [code](/examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Inpaint.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Inpaint.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Blockwise-ControlNet-Inpaint.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Inpaint.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Blockwise-ControlNet-Inpaint.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Inpaint.py) |
| [DiffSynth-Studio/Qwen-Image-In-Context-Control-Union](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-In-Context-Control-Union) | [code](/examples/qwen_image/model_inference/Qwen-Image-In-Context-Control-Union.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-In-Context-Control-Union.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-In-Context-Control-Union.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-In-Context-Control-Union.py) |
| [DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit-Lowres-Fix.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-Lowres-Fix.py) | - | - | - | - |
|[DiffSynth-Studio/Qwen-Image-i2L](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-i2L)|[code](/examples/qwen_image/model_inference/Qwen-Image-i2L.py)|[code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-i2L.py)|-|-|-|-|

Special Training Scripts:

* Differential LoRA Training: [doc](/docs/en/Training/Differential_LoRA.md), [code](/examples/qwen_image/model_training/special/differential_training/)
* FP8 Precision Training: [doc](/docs/en/Training/FP8_Precision.md), [code](/examples/qwen_image/model_training/special/fp8_training/)
* Two-stage Split Training: [doc](/docs/en/Training/Split_Training.md), [code](/examples/qwen_image/model_training/special/split_training/)
* End-to-end Direct Distillation: [doc](/docs/en/Training/Direct_Distill.md), [code](/examples/qwen_image/model_training/lora/Qwen-Image-Distill-LoRA.sh)

DeepSpeed ZeRO Stage 3 Training: The Qwen-Image series models support DeepSpeed ZeRO Stage 3 training, which partitions the model across multiple GPUs. Taking full parameter training of the Qwen-Image model as an example, the following modifications are required:

* `--config_file examples/qwen_image/model_training/full/accelerate_config_zero3.yaml`
* `--initialize_model_on_cpu`

## Model Inference

Models are loaded via `QwenImagePipeline.from_pretrained`, see [Loading Models](/docs/en/Pipeline_Usage/Model_Inference.md#loading-models).

Input parameters for `QwenImagePipeline` inference include:

* `prompt`: Prompt describing the content appearing in the image.
* `negative_prompt`: Negative prompt describing content that should not appear in the image, default value is `""`.
* `cfg_scale`: Classifier-free guidance parameter, default value is 4. When set to 1, it no longer takes effect.
* `input_image`: Input image for image-to-image generation, used in conjunction with `denoising_strength`.
* `denoising_strength`: Denoising strength, range is 0~1, default value is 1. When the value approaches 0, the generated image is similar to the input image; when the value approaches 1, the generated image differs more from the input image. When `input_image` parameter is not provided, do not set this to a non-1 value.
* `inpaint_mask`: Image inpainting mask image.
* `inpaint_blur_size`: Edge blur width for image inpainting.
* `inpaint_blur_sigma`: Edge blur strength for image inpainting.
* `height`: Image height, must be a multiple of 16.
* `width`: Image width, must be a multiple of 16.
* `seed`: Random seed. Default is `None`, meaning completely random.
* `rand_device`: Computing device for generating random Gaussian noise matrix, default is `"cpu"`. When set to `cuda`, different GPUs will produce different generation results.
* `num_inference_steps`: Number of inference steps, default value is 30.
* `exponential_shift_mu`: Fixed parameter used in sampling timesteps. Leave blank to sample based on image width and height.
* `blockwise_controlnet_inputs`: Blockwise ControlNet model inputs.
* `eligen_entity_prompts`: EliGen partition control prompts.
* `eligen_entity_masks`: EliGen partition control region mask images.
* `eligen_enable_on_negative`: Whether to enable EliGen partition control on the negative side of CFG.
* `edit_image`: Edit model images to be edited, supports multiple images.
* `edit_image_auto_resize`: Whether to automatically scale edit images.
* `edit_rope_interpolation`: Whether to enable ROPE interpolation on low-resolution edit images.
* `context_image`: In-Context Control input image.
* `tiled`: Whether to enable VAE tiling inference, default is `False`. Setting to `True` can significantly reduce VRAM usage during VAE encoding/decoding stages, producing slight errors and slightly longer inference time.
* `tile_size`: Tile size during VAE encoding/decoding stages, default is 128, only effective when `tiled=True`.
* `tile_stride`: Tile stride during VAE encoding/decoding stages, default is 64, only effective when `tiled=True`, must be less than or equal to `tile_size`.
* `progress_bar_cmd`: Progress bar, default is `tqdm.tqdm`. Can be disabled by setting to `lambda x:x`.

If VRAM is insufficient, please enable [VRAM Management](/docs/en/Pipeline_Usage/VRAM_management.md). We provide recommended low VRAM configurations for each model in the example code, see the table in the "Model Overview" section above.

## Model Training

Qwen-Image series models are uniformly trained through [`examples/qwen_image/model_training/train.py`](/examples/qwen_image/model_training/train.py), and the script parameters include:

* General Training Parameters
    * Dataset Basic Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Metadata file path of the dataset.
        * `--dataset_repeat`: Number of times the dataset is repeated in each epoch.
        * `--dataset_num_workers`: Number of processes for each DataLoader.
        * `--data_file_keys`: Field names to be loaded from metadata, usually image or video file paths, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths of models to be loaded. JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, e.g., `"Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors"`. Separated by commas.
        * `--extra_inputs`: Extra input parameters required by the model Pipeline, e.g., extra parameters `edit_image` when training image editing model Qwen-Image-Edit, separated by `,`.
        * `--fp8_models`: Models loaded in FP8 format, consistent with `--model_paths` or `--model_id_with_origin_paths` format. Currently only supports models whose parameters are not updated by gradients (no gradient backpropagation, or gradients only update their LoRA).
    * Training Basic Configuration
        * `--learning_rate`: Learning rate.
        * `--num_epochs`: Number of epochs.
        * `--trainable_models`: Trainable models, e.g., `dit`, `vae`, `text_encoder`.
        * `--find_unused_parameters`: Whether there are unused parameters in DDP training. Some models contain redundant parameters that do not participate in gradient calculation, and this setting needs to be enabled to avoid errors in multi-GPU training.
        * `--weight_decay`: Weight decay size, see [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
        * `--task`: Training task, default is `sft`. Some models support more training modes, please refer to the documentation of each specific model.
    * Output Configuration
        * `--output_path`: Model saving path.
        * `--remove_prefix_in_ckpt`: Remove prefix in the state dict of the model file.
        * `--save_steps`: Interval of training steps to save the model. If this parameter is left blank, the model is saved once per epoch.
    * LoRA Configuration
        * `--lora_base_model`: Which model to add LoRA to.
        * `--lora_target_modules`: Which layers to add LoRA to.
        * `--lora_rank`: Rank of LoRA.
        * `--lora_checkpoint`: Path of the LoRA checkpoint. If this path is provided, LoRA will be loaded from this checkpoint.
        * `--preset_lora_path`: Preset LoRA checkpoint path. If this path is provided, this LoRA will be loaded in the form of being merged into the base model. This parameter is used for LoRA differential training.
        * `--preset_lora_model`: Model that the preset LoRA is merged into, e.g., `dit`.
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to memory.
        * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
    * Image Width/Height Configuration (Applicable to Image Generation and Video Generation Models)
        * `--height`: Height of image or video. Leave `height` and `width` blank to enable dynamic resolution.
        * `--width`: Width of image or video. Leave `height` and `width` blank to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area of image or video frames. When dynamic resolution is enabled, images with resolution larger than this value will be downscaled, and images with resolution smaller than this value will remain unchanged.
* Qwen-Image Specific Parameters
    * `--tokenizer_path`: Path of the tokenizer, applicable to text-to-image models, leave blank to automatically download from remote.
    * `--processor_path`: Path of the processor, applicable to image editing models, leave blank to automatically download from remote.

We have built a sample image dataset for your testing. You can download this dataset with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/example_image_dataset --local_dir ./data/example_image_dataset
```

We have written recommended training scripts for each model, please refer to the table in the "Model Overview" section above. For how to write model training scripts, please refer to [Model Training](/docs/en/Pipeline_Usage/Model_Training.md); for more advanced training algorithms, please refer to [Training Framework Detailed Explanation](/docs/Training/).