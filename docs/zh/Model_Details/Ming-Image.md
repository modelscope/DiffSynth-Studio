# Ming-Image-0.2-Design

Ming-Image-0.2-Design 是蚂蚁 inclusionAI 的视觉设计生成模型（UI/海报/信息图/文字密集排版），6B DiT + BailingMM2 MLLM 条件编码，支持透明背景 RGBA 输出与图像编辑。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [inclusionAI/Ming-Image-0.2-Design](https://www.modelscope.cn/models/inclusionAI/Ming-Image-0.2-Design) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 3G 显存即可运行。

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

## 模型总览

|模型 ID|推理|低显存推理|并行推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|-|
|[inclusionAI/Ming-Image-0.2-Design](https://www.modelscope.cn/models/inclusionAI/Ming-Image-0.2-Design)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_inference/Ming-Image-0.2-Design.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_inference_low_vram/Ming-Image-0.2-Design.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_inference_parallel/Ming-Image-0.2-Design.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_training/full/Ming-Image-0.2-Design.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_training/validate_full/Ming-Image-0.2-Design.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_training/lora/Ming-Image-0.2-Design.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ming_image/model_training/validate_lora/Ming-Image-0.2-Design.py)|

## 模型推理

模型通过 `MingImagePipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`MingImagePipeline` 推理的输入参数包括：

* `prompt`: 提示词，支持自然语言描述与结构化 Figma 风格 JSON 排版。
* `cfg_scale`: Classifier-free guidance 强度，默认 `1.0`（不启用 CFG）。
* `height`: 图像高度，默认 `1024`。
* `width`: 图像宽度，默认 `1024`。
* `seed`: 随机种子，默认 `None`。
* `rand_device`: 随机数生成设备，默认 `"cpu"`。
* `num_inference_steps`: 去噪步数，默认 `12`。
* `edit_image`: 编辑参考图（PIL Image），传入后进入图像编辑模式，默认 `None`。
* `tiled`: 是否启用 VAE 分块解码，默认 `False`。
* `tile_size`: VAE 分块大小，默认 `128`。
* `tile_stride`: VAE 分块步长，默认 `64`。
* `progress_bar_cmd`: 进度条命令，默认 `tqdm`。

模型输出为 RGBA 四通道图像（支持透明背景）。显存不足时，请参考[显存管理](../Pipeline_Usage/VRAM_Management.md)启用 offload 与量化。

## 模型训练

ming_image 系列模型统一通过 `examples/ming_image/model_training/train.py` 进行训练，脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloader 的进程数量。
        * `--data_file_keys`: 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 `,` 分隔。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID。用逗号分隔。
        * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，以 `,` 分隔。
        * `--fp8_models`: 以 FP8 格式加载的模型，目前仅支持参数不被梯度更新的模型。
        * `--quant_options`：对加载的模型进行动态量化。以 `;` 分隔多个条目，每个为 `<模型字符串>:<method>[/<exclude_modules>]`，`<模型字符串>` 需与 `--model_paths`/`--model_id_with_origin_paths` 中的一致，`method` 为已注册的量化方法（如 `bitsandbytes_nf4`），`exclude_modules` 为可选的保持全精度的层。
    * 训练基础配置
        * `--learning_rate`: 学习率。
        * `--num_epochs`: 轮数（Epoch）。
        * `--trainable_models`: 可训练的模型，例如 `dit`、`vae`、`text_encoder`。
        * `--find_unused_parameters`: DDP 训练中是否存在未使用的参数。
        * `--weight_decay`: 权重衰减大小。
        * `--task`: 训练任务，默认为 `sft`。
    * 输出配置
        * `--output_path`: 模型保存路径。
        * `--remove_prefix_in_ckpt`: 在模型文件的 state dict 中移除前缀。
        * `--save_steps`: 保存模型的训练步数间隔。
    * LoRA 配置
        * `--lora_base_model`: LoRA 添加到哪个模型上。
        * `--lora_target_modules`: LoRA 添加到哪些层上。
        * `--lora_rank`: LoRA 的秩（Rank）。
        * `--lora_checkpoint`: LoRA 检查点的路径。
        * `--preset_lora_path`: 预置 LoRA 检查点路径，用于 LoRA 差分训练。
        * `--preset_lora_model`: 预置 LoRA 融入的模型，例如 `dit`。
    * 梯度配置
        * `--use_gradient_checkpointing`: 是否启用 gradient checkpointing。
        * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。
        * `--gradient_accumulation_steps`: 梯度累积步数。
    * 分辨率配置
        * `--height`: 图像/视频的高度。留空启用动态分辨率。
        * `--width`: 图像/视频的宽度。留空启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
        * `--num_frames`: 视频的帧数（仅视频生成模型）。
* Ming-Image-0.2-Design 专有参数
    * `--tokenizer_path`: tokenizer 路径，留空时默认从模型的 `mllm/` 目录加载。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型。

样例数据集下载命令：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。