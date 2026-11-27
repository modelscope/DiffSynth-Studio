# Ming-Image-0.2-Design

Ming-Image-0.2-Design is a visual design generation model (UI/poster/infographic/text-heavy layout) from Ant inclusionAI, with a 6B DiT + BailingMM2 MLLM condition encoder, supporting transparent-background RGBA output and image editing.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [inclusionAI/Ming-Image-0.2-Design](https://www.modelscope.cn/models/inclusionAI/Ming-Image-0.2-Design) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 3GB VRAM.

```python
from diffsynth.pipelines.ming_image import MingImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from PIL import Image
import torch

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
    "quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw"),
}
disk_vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MingImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **disk_vram_config),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="mllm/model*.safetensors", **disk_vram_config),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="connector/model*.safetensors", **disk_vram_config),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="mlp/model.safetensors", **disk_vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="mllm/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Text-to-Image with a natural-language prompt
prompt = """Landing page for a productivity app called Flow. Clean minimal design, white background, generous whitespace. Top navigation with the wordmark 'Flow' on the left and links Product, Pricing, Docs on the right. Centered hero with a bold headline 'Focus without the noise' and a gray subheading 'One calm workspace to plan, focus, and finish.' A single purple call-to-action button labeled 'Start free'. Below the hero, three feature cards with small icons, soft rounded corners and soft shadows. Card 1: title 'Plan your day', body text 'Turn scattered tasks into a clear, realistic plan.' Card 2: title 'Focus sessions', body text 'Start a distraction-free session and stay in flow.' Card 3: title 'See progress', body text 'Know what moved, what slipped, and what comes next.' Modern sans-serif typography. Keep every heading and body line short and legible. Render all text exactly as written. Do not invent additional copy."""
image = pipe(prompt=prompt, height=2048, width=2048, seed=0, num_inference_steps=12, cfg_scale=1.0)
image.save("image1.png")

# Text-to-Image with a structured Figma-style JSON prompt
prompt = """{
  "canvas_settings": {
    "aspect_ratio": "1:1, 2048 × 2048 px",
    "ambient_lighting": "Four season-specific lighting conditions unified by one fixed eye-level sun direction, consistent horizon at 48% canvas height, matching atmospheric depth, and natural cinematic exposure.",
    "image_style": "High-resolution cinematic photorealistic environmental concept sheet; four equal edge-to-edge vertical panels with crisp shared boundaries and no gutters, frames, extra title, or decorative inter-panel objects. Fixed 35 mm lens, eye-level three-quarter front-right camera angle, identical perspective and cabin registration in every panel. Each panel uses restrained bold uppercase condensed sans-serif labeling at the bottom."
  },
  "layers": [
    {
      "description": "Leftmost spring panel. The same small one-story rectangular log cabin appears at a fixed local position and scale: horizontal dark-brown logs, pale stone foundation, centered vertical-plank front door, two square four-pane front windows, one matching right-side window, and a charcoal standing-seam gabled roof with identical ridge height and overhang. Fresh pale-green grass, budding branches, white and pink blossoms, rain-darkened soil, shallow puddles, wet roof highlights, fine recent-rain droplets, and soft mist fill the landscape beneath an overcast clearing sky. Render the exact label \"SPRING\" centered near the bottom in bold condensed uppercase white sans-serif with a subtle dark shadow.",
      "coordinates": "cx: 0.125, cy: 0.500, w: 0.250, h: 1",
      "hierarchy_and_relation": "Complete first panel; aligned to the shared horizon and fixed cabin registration, flush with the canvas left edge and directly abutting the second panel without a gutter.",
      "color_specs": ["#A9C7D2", "#D8E4DE", "#7FAF69", "#B7D58B", "#F2D7DE", "#F4F1E8", "#5A3B28", "#2C3032", "#7B7468", "#FFFFFF"]
    },
    {
      "description": "Second summer panel. Repeat the cabin with exactly the same architecture, camera angle, local placement, dimensions, centered plank door, front and side windows, stone foundation, roof pitch, ridge height, and overhang. Surround it with lush meadow grass, dense layered deciduous foliage, full tree canopies, small sunlit shrubs, and dry natural ground; use a bright clear sky, strong warm sunlight from the same direction, crisp leaf highlights, controlled shadows, and mild atmospheric haze. Render the exact label \"SUMMER\" centered near the bottom in bold condensed uppercase white sans-serif with a subtle dark shadow.",
      "coordinates": "cx: 0.375, cy: 0.500, w: 0.250, h: 1",
      "hierarchy_and_relation": "Complete second panel; aligned to the shared horizon and fixed cabin registration, directly abutting the first and third panels without gutters.",
      "color_specs": ["#4FA9D8", "#BDE8F2", "#26733E", "#4F9A45", "#79B84A", "#D8C66A", "#5A3B28", "#2C3032", "#7B7468", "#FFFFFF"]
    },
    {
      "description": "Third autumn panel. Repeat the cabin with exactly the same architecture, camera angle, local placement, dimensions, centered plank door, front and side windows, stone foundation, roof pitch, ridge height, and overhang. Cover the landscape with amber grass, rust-orange and golden deciduous foliage, scattered fallen leaves, and a few leaves drifting naturally in the air. Use warm late-afternoon sunlight from the same direction, long soft shadows, copper rim light on logs and roof, a pale warm sky, and subtle atmospheric depth. Render the exact label \"AUTUMN\" centered near the bottom in bold condensed uppercase white sans-serif with a subtle dark shadow.",
      "coordinates": "cx: 0.625, cy: 0.500, w: 0.250, h: 1",
      "hierarchy_and_relation": "Complete third panel; aligned to the shared horizon and fixed cabin registration, directly abutting the second and fourth panels without gutters.",
      "color_specs": ["#E6B06A", "#F0D6A2", "#D66A2C", "#A84324", "#E2A62B", "#80613B", "#5A3B28", "#2C3032", "#7B7468", "#FFFFFF"]
    },
    {
      "description": "Rightmost winter panel. Repeat the cabin with exactly the same architecture, camera angle, local placement, dimensions, centered plank door, front and side windows, stone foundation, roof pitch, ridge height, and overhang. Blanket the ground and roof with natural snow accumulation while keeping doors and windows readable; add bare deciduous branches, snow-laden low shrubs, faint footprints away from the cabin, and sparse fine snowfall. Use cool blue ambient light, a pale clouded sky, soft low-contrast shadows from the same direction, icy edge highlights, and slight winter haze. Render the exact label \"WINTER\" centered near the bottom in bold condensed uppercase white sans-serif with a subtle dark shadow.",
      "coordinates": "cx: 0.875, cy: 0.500, w: 0.250, h: 1",
      "hierarchy_and_relation": "Complete fourth panel; aligned to the shared horizon and fixed cabin registration, directly abutting the third panel without a gutter and flush with the canvas right edge.",
      "color_specs": ["#AFC8DE", "#DCE8F1", "#F4F7F8", "#C6D8E5", "#8198AA", "#6D5949", "#5A3B28", "#2C3032", "#7B7468", "#FFFFFF"]
    }
  ]
}"""
image = pipe(prompt=prompt, height=2048, width=2048, seed=0, num_inference_steps=12, cfg_scale=1.0)
image.save("image2.png")

# Image Editing
prompt = "Change the app's color scheme from purple to green: the call-to-action button, icons and accents become green."
image = pipe(prompt=prompt, edit_image=Image.open("image1.png"), height=2048, width=2048, seed=1, num_inference_steps=12, cfg_scale=1.0)
image.save("image3.png")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Parallel Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|-|
|[inclusionAI/Ming-Image-0.2-Design](https://www.modelscope.cn/models/inclusionAI/Ming-Image-0.2-Design)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_inference/Ming-Image-0.2-Design.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_inference_low_vram/Ming-Image-0.2-Design.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_inference_parallel/Ming-Image-0.2-Design.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_training/full/Ming-Image-0.2-Design.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_training/validate_full/Ming-Image-0.2-Design.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_training/lora/Ming-Image-0.2-Design.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_training/validate_lora/Ming-Image-0.2-Design.py)|

## Model Inference

The model is loaded via `MingImagePipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `MingImagePipeline` inference include:

* `prompt`: Prompt, supporting natural-language descriptions and structured Figma-style JSON layouts.
* `cfg_scale`: Classifier-free guidance scale, default `1.0` (CFG disabled).
* `height`: Image height, default `1024`.
* `width`: Image width, default `1024`.
* `seed`: Random seed, default `None`.
* `rand_device`: Device for random number generation, default `"cpu"`.
* `num_inference_steps`: Number of denoising steps, default `12`.
* `edit_image`: Reference image for editing (PIL Image); when provided, enters image editing mode, default `None`.
* `tiled`: Whether to enable tiled VAE decoding, default `False`.
* `tile_size`: VAE tile size, default `128`.
* `tile_stride`: VAE tile stride, default `64`.
* `progress_bar_cmd`: Progress bar command, default `tqdm`.

The model outputs RGBA four-channel images (transparent background supported). When VRAM is insufficient, refer to [VRAM Management](../Pipeline_Usage/VRAM_Management.md) to enable offload and quantization.

## Model Training

Models in the ming_image series are trained uniformly via `examples/ming_image/model_training/train.py`. The script parameters include:

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
* Ming-Image-0.2-Design Specific Parameters
    * `--tokenizer_path`: Path to the tokenizer; when empty, loads from the model's `mllm/` directory by default.
    * `--initialize_model_on_cpu`: Whether to initialize models on CPU.

Example dataset download command:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).