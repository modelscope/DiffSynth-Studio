# SenseNova-U1

SenseNova-U1 is a unified multimodal model series open-sourced by SenseTime. It adopts a Mixture-of-Transformers (MoT) architecture that keeps an understanding branch and a generation branch in parallel within the same Transformer layers, and performs flow matching denoising directly in pixel space without a separate VAE or text encoder.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [SenseNova/SenseNova-U1.5-8B-MoT](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 4GB VRAM.

```python
from diffsynth.pipelines.sensenova_u1_image import SenseNovaU1ImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = SenseNovaU1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="model*.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="./"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = "A neon bar sign that clearly reads \"OPEN LATE\", dark interior, moody reflections, easy text rendering. Any text in the image must be rendered exactly as written in quotation marks, with correct spelling, clean typography, and strong readability."
image = pipe(prompt=prompt, seed=42, height=2048, width=2048, num_inference_steps=50, cfg_scale=4.0, shift=3.0)
image.save("image_SenseNova-U1.5-8B-MoT.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[SenseNova/SenseNova-U1.5-8B-MoT: T2I](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT.py)|
|[SenseNova/SenseNova-U1.5-8B-MoT: Edit](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT-Edit.py)|
|[SenseNova/SenseNova-U1.5-8B-MoT-LoRAs: 8-step](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-LoRAs)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-LoRA-8step.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-LoRA-8step.py)|-|-|-|-|
|[SenseNova/SenseNova-U1.5-8B-MoT-SFT: T2I](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-SFT)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT-SFT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT-SFT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT-SFT.py)|
|[SenseNova/SenseNova-U1.5-8B-MoT-SFT: Edit](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-SFT)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-SFT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-SFT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT-SFT-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT-SFT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT-SFT-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT-SFT-Edit.py)|

## Model Inference

The model is loaded via `SenseNovaU1ImagePipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `SenseNovaU1ImagePipeline` inference include:

* `prompt`: Text prompt. Acts as the editing instruction in image editing mode.
* `cfg_scale`: Classifier-Free Guidance scale, defaults to 4.0. The reference hardcodes the unconditional prefix text and exposes no negative prompt, so this pipeline has no `negative_prompt` parameter either.
* `height`: Output image height, defaults to 2048. Must be a multiple of 32.
* `width`: Output image width, defaults to 2048. Must be a multiple of 32.
* `seed`: Random seed, defaults to random.
* `rand_device`: Device for noise generation, defaults to `"cuda"`.
* `num_inference_steps`: Number of inference steps, defaults to 50.
* `shift`: Timestep shift affecting sigma computation, defaults to 3.0.
* `think_mode`: Whether the model writes a reasoning block before generating, defaults to False.
* `edit_image`: Input image, either a single `Image.Image` or a list of images. Passing it switches to image editing mode, defaults to None (text-to-image mode).
* `input_image`: Target image supplied during training, not needed for inference.

> **VRAM note**: SenseNova-U1.5-8B-MoT has about 17.5B parameters, requiring roughly 35GB of VRAM for resident BF16 weights. When generating 2048x2048 images, enabling VRAM management (vram_config) or using the low VRAM inference script is recommended, see [Memory Management](../Pipeline_Usage/Model_Inference.md).

### Image Editing

Passing `edit_image` switches to image editing mode. The input images are encoded by the understanding branch's vision encoder and spliced into the conditioning prefix. The negative branch carries the input images without the editing instruction, so guidance points away from "the input image unchanged" rather than away from "any image".

The output resolution must be specified explicitly; it is not derived from the input image:

```python
from PIL import Image

edit_image = Image.open("input.jpg").convert("RGB")
image = pipe(prompt="Change the dress to pink.", edit_image=edit_image, height=2048, width=2048, seed=42)
```

Passing a list of images performs multi-image editing. The images are numbered in the order given,
so the prompt can refer to them as Figure 1, Figure 2, and so on:

```python
image = pipe(
    prompt="Change the color of the dress in Figure 1 to the color shown in Figure 2.",
    edit_image=[edit_image, color_image],
    height=2048, width=2048, seed=42,
)
```

### Think Mode

With `think_mode=True` the model first autoregressively writes a reasoning block — planning the
composition, environment, and lighting — and then generates the image from it. The reasoning text
shapes the image but is not returned:

```python
image = pipe(prompt="A neon bar sign that clearly reads \"OPEN LATE\"", think_mode=True, seed=42)
image.save("image.jpg")
```

Decoding is greedy (no temperature or nucleus sampling), capped at 1024 tokens, and stops at
`</think>` or `<|im_end|>`. Only the conditional branch reasons; the unconditional branch is unchanged.

> Each reasoning token passes through `lm_head` (about 0.6B parameters). Combined with disk
> offloading this becomes the bottleneck, so the low VRAM scripts do not offer this mode.

### Fast Inference (8-step LoRA)

The official distillation LoRA cuts denoising from 50 steps to 8 and allows `cfg_scale` of 1.0. The
latter makes the framework skip the negative branch, removing one 17.5B forward pass per step, for
roughly a 12x speedup overall:

```python
pipe.load_lora(pipe.dit, ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT-LoRAs", origin_file_pattern="SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors"))
image = pipe(prompt=prompt, seed=42, height=2048, width=2048, num_inference_steps=8, cfg_scale=1.0, shift=3.0)
```

The LoRA targets the generation branch's attention and MLP layers (42 layers x 7 modules, 294 in
total); the base weights are unchanged.

### SFT Checkpoint

`SenseNova-U1.5-8B-MoT-SFT` is architecturally identical to the model above and differs only in
training stage: it is the checkpoint after Unified SFT, whereas the released model additionally goes
through Multi-Expert RL and MOPD training. Use the released model for generation quality; the SFT
checkpoint is available as a fine-tuning starting point or for ablation, and only the `model_id`
changes.

### Architecture Notes

SenseNova-U1 differs from typical diffusion models in three ways that are worth knowing when using it:

* **MoT branches share layers**: each of the 42 decoder layers holds weights for both the understanding branch and the generation branch, selected per token. The two branches cannot be split into separate models, which is why DiffSynth registers a single `sensenova_u1_dit` component.
* **Pixel-space denoising without a VAE**: `latents` is a `(1, 3, H, W)` pixel tensor throughout, and the final output is restored to full resolution by a pixel head (a PixelShuffle convolutional decoder) rather than a VAE decode.
* **Conditioning through a prefix KV cache**: there is no separate text encoder. The prompt is first encoded by the understanding branch into `past_key_values`, and every denoising step then runs the image tokens through the generation branch against that cache.

## Model Training

Models in the sensenova_u1 series are trained uniformly via `examples/sensenova_u1/model_training/train.py`. The script parameters include:

* General Training Parameters
    * Dataset Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Path to the dataset metadata file.
        * `--dataset_repeat`: Number of dataset repeats per epoch.
        * `--dataset_num_workers`: Number of processes per DataLoader.
        * `--data_file_keys`: Field names to load from metadata, typically paths to image or video files, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths to load models from, in JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, separated by commas.
        * `--extra_inputs`: Additional input parameters required by the model Pipeline, separated by `,`.
        * `--fp8_models`: Models to load in FP8 format, currently only supported for models whose parameters are not updated by gradients.
        * `--quant_options`: Dynamically quantize loaded models. Semicolon-separated entries, each `<model_string>:<method>[/<exclude_modules>]`, where `<model_string>` matches an entry in `--model_paths`/`--model_id_with_origin_paths`, `method` is a registered method (e.g. `bitsandbytes_nf4`), and `exclude_modules` optionally lists layers kept in full precision.
    * Basic Training Configuration
        * `--learning_rate`: Learning rate.
        * `--num_epochs`: Number of epochs.
        * `--trainable_models`: Trainable models, e.g., `dit`, `vae`, `text_encoder`.
        * `--find_unused_parameters`: Whether unused parameters exist in DDP training.
        * `--weight_decay`: Weight decay magnitude.
        * `--task`: Training task, defaults to `sft`.
    * Output Configuration
        * `--output_path`: Path to save the model.
        * `--remove_prefix_in_ckpt`: Remove prefix in the model's state dict.
        * `--save_steps`: Interval in training steps to save the model.
    * LoRA Configuration
        * `--lora_base_model`: Which model to add LoRA to.
        * `--lora_target_modules`: Which layers to add LoRA to.
        * `--lora_rank`: Rank of LoRA.
        * `--lora_checkpoint`: Path to LoRA checkpoint.
        * `--preset_lora_path`: Path to preset LoRA checkpoint for LoRA differential training.
        * `--preset_lora_model`: Which model to integrate preset LoRA into, e.g., `dit`.
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.
        * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
    * Resolution Configuration
        * `--height`: Height of the image/video. Leave empty to enable dynamic resolution.
        * `--width`: Width of the image/video. Leave empty to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area, images larger than this will be scaled down during dynamic resolution.
        * `--num_frames`: Number of frames for video (video generation models only).
* SenseNova-U1 Specific Parameters
    * `--tokenizer_config`: Path to the tokenizer config, used to load Qwen2Tokenizer for text tokenization.
    * `--initialize_model_on_cpu`: Whether to initialize models on CPU, which lowers peak GPU VRAM usage.

LoRA training is recommended to target the generation branch only, i.e. spell out the `_mot_gen` suffix in `--lora_target_modules`: `q_proj_mot_gen,k_proj_mot_gen,v_proj_mot_gen,o_proj_mot_gen,mlp_mot_gen.gate_proj,mlp_mot_gen.up_proj,mlp_mot_gen.down_proj`. peft matches by suffix, so writing only `gate_proj` would also match the understanding branch's `mlp.gate_proj`.

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
