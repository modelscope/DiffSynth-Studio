# FLUX.3

FLUX.3 Image Klein is a 5.6B open-weight image generation model from Black Forest Labs that supports text-to-image generation, editing with up to 10 reference images, and bounding-box layout control in one model.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Model Access

`black-forest-labs/flux-3-image-klein` and `black-forest-labs/flux-3-image-klein-base` are access-request-only models on ModelScope. Before downloading them:

1. Request access on the model page ([Klein](https://modelscope.cn/models/black-forest-labs/flux-3-image-klein) or [Klein Base](https://modelscope.cn/models/black-forest-labs/flux-3-image-klein-base)).
2. Log in via the CLI with your ModelScope token: `modelscope login --token ms-xxx`. Get your token at [Settings / Access Token](https://modelscope.cn/my/settings/token).

Access must be granted for the same account used to log in.

## Quick Start

Running the following code will load the [black-forest-labs/flux-3-image-klein](https://www.modelscope.cn/models/black-forest-labs/flux-3-image-klein) model for inference. VRAM management is enabled, and the framework automatically controls parameter loading based on available VRAM. Minimum 4GB VRAM is required to run.

```python
from PIL import Image
import torch

from diffsynth.pipelines.flux3_image import Flux3ImagePipeline, ModelConfig


vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = Flux3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / 1024**3 - 0.5,
)

prompt = "Masterpiece, best quality. Cinematic photograph of a beautiful young woman with long black hair wearing a navy sailor-style school uniform with a green necktie, standing in a sunlit school corridor with large windows, warm afternoon light and soft bokeh in the background."
image_1 = pipe(prompt=prompt, seed=0, num_inference_steps=16, cfg_scale=1.0)
image_1.save("image1.png")

prompt = (
    "Masterpiece, best quality. Cinematic photograph of <girl_1>, a cheerful young woman with a short auburn bob "
    "wearing a white short-sleeved blouse and a deep red pleated skirt with red Mary Jane shoes, standing on a leafy campus path at golden hour, "
    "a brown leather satchel <bag_1> resting on the ground at frame-left and a vintage bicycle <bicycle_1> parked at frame-right, soft bokeh in the background. "
    '[{"id":"girl_1","bbox":[60,330,980,690],"desc":"Cheerful young woman with a short auburn bob, a white short-sleeved blouse and a deep red pleated skirt with red Mary Jane shoes, standing facing the camera."},'
    '{"id":"bag_1","bbox":[740,100,940,360],"desc":"Brown leather school satchel with brass buckles and a shoulder strap, resting on the leaf-covered ground."},'
    '{"id":"bicycle_1","bbox":[380,720,880,980],"desc":"Vintage bicycle with a dark green frame, cream tires and a woven front basket, leaning on its kickstand."}]'
)
image_2 = pipe(prompt=prompt, seed=1, num_inference_steps=16, cfg_scale=1.0)
image_2.save("image2.png")

prompt = "Masterpiece, best quality. Cinematic photograph of the two young women from the two reference images standing side by side at the same scale in a sunlit school courtyard, the long-black-haired woman in the navy sailor school uniform and the short-auburn-haired woman in the white blouse and deep red pleated skirt, warm afternoon light, keeping both women's hairstyles and outfits unchanged."
edit_image = [Image.open("image1.png"), Image.open("image2.png")]
image_3 = pipe(prompt=prompt, edit_image=edit_image, seed=2, num_inference_steps=16, cfg_scale=1.0)
image_3.save("image3.png")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[black-forest-labs/flux-3-image-klein](https://www.modelscope.cn/models/black-forest-labs/flux-3-image-klein)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference/flux-3-image-klein.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference_low_vram/flux-3-image-klein.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/full/flux-3-image-klein.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/validate_full/flux-3-image-klein.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/lora/flux-3-image-klein.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/validate_lora/flux-3-image-klein.py)|
|[black-forest-labs/flux-3-image-klein-base](https://www.modelscope.cn/models/black-forest-labs/flux-3-image-klein-base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference/flux-3-image-klein-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference_low_vram/flux-3-image-klein-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/full/flux-3-image-klein-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/validate_full/flux-3-image-klein-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/lora/flux-3-image-klein-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/validate_lora/flux-3-image-klein-base.py)|
|[DiffSynth-Studio/flux-3-image-klein-entropack](https://www.modelscope.cn/models/DiffSynth-Studio/flux-3-image-klein-entropack)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference/flux-3-image-klein-EntroPack.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference_low_vram/flux-3-image-klein-EntroPack.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/lora/flux-3-image-klein-entropack.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/validate_lora/flux-3-image-klein-entropack.py)|

## Model Inference

The model is loaded via `Flux3ImagePipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `Flux3ImagePipeline` inference include:

* `prompt`: Prompt describing the target image, edit, or layout; defaults to `""`.
* `negative_prompt`: Negative prompt used by Base when CFG is enabled; defaults to `""`.
* `cfg_scale`: Classifier-free guidance strength, defaults to 1.0. Distilled requires 1.0; the Base examples use 3.5.
* `input_image`: Input image for image-to-image generation; generation starts from pure noise when omitted.
* `denoising_strength`: Image-to-image denoising strength from 0 to 1, defaults to 1.0.
* `edit_image`: Reference image(s) for editing, accepting one `PIL.Image` or a list of images; up to 10 references are supported.
* `height`: Output image height, defaults to 1024 and is aligned to a multiple of 16.
* `width`: Output image width, defaults to 1024 and is aligned to a multiple of 16.
* `seed`: Random seed, defaults to `None`.
* `rand_device`: Device used to generate the initial Gaussian noise, defaults to `"cpu"`.
* `num_inference_steps`: Number of inference steps, defaults to 16. The Distilled examples use 16 steps and the Base examples use 50.
* `alpha`: Rational-shift coefficient for the FLUX.3 flow-matching schedule, defaults to 6.93.
* `progress_bar_cmd`: Function wrapping the inference timesteps with a progress bar, defaults to `tqdm`.

If VRAM is insufficient, please enable [VRAM Management](../Pipeline_Usage/VRAM_management.md). Recommended configurations are linked in the "Model Overview" table above.

## Model Training

Models in the FLUX.3 series are trained uniformly via [`examples/flux3/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/train.py). The script parameters include:

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
* FLUX.3 Specific Parameters
    * `--processor_path`: Local processor path or model ID; when omitted, the Base model's `text_encoder/` is used.
    * `--initialize_model_on_cpu`: Initialize models on CPU to reduce peak VRAM during distributed training startup.

We have built a sample image dataset for your testing. You can download this dataset with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
