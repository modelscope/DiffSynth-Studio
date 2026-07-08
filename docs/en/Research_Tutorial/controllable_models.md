# ModelScope AIGC Series Course - Controllable Generation Technology

This experiment uses **Diffusion-Templates** as the framework to systematically introduce various controllable generation techniques for image generation models, and demonstrate how to train a controllable generation module from scratch.

Related Resources:

* Open Source Code: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* Technical Report: [arXiv](https://arxiv.org/abs/2604.24351)
* Project Homepage: [GitHub](https://modelscope.github.io/diffusion-templates-web/)
* Documentation: [English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)、[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)
* Online Demo: [ModelScope Studio](https://modelscope.cn/studios/DiffSynth-Studio/Diffusion-Templates)
* Model Collection: [ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/KleinBase4B-Templates)、[ModelScope International](https://modelscope.ai/collections/DiffSynth-Studio/KleinBase4B-Templates)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/kleinbase4b-templates)
* Dataset: [ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[ModelScope International](https://modelscope.ai/collections/DiffSynth-Studio/ImagePulseV2)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/imagepulsev2)

```python
!pip install diffsynth==2.0.15 transformers==5.8.1
```

```python
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch
from modelscope import dataset_snapshot_download, snapshot_download
from PIL import Image
import numpy as np

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

def show_images(images, resolution):
    images = [i.resize((resolution, resolution)).convert("RGB") for i in images]
    images = [np.array(i) for i in images]
    images = np.concat(images, axis=1)
    images = Image.fromarray(images)
    return images
```

## Image Structure Control

First, load the base model [black-forest-labs/FLUX.2-klein-base-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B). This is a 4B-parameter image generation model, and all controllable generation modules in this experiment will be mounted on top of this base model.

```python
pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
```

[ControlNet](https://arxiv.org/abs/2302.05543) is one of the earliest controllable generation techniques for Diffusion models. It uses structural conditions such as **depth maps, edge maps, and pose maps** to achieve **pixel-level** control over the generated image.

By loading [DiffSynth-Studio/Template-KleinBase4B-ControlNet](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-ControlNet) in Template format, you can generate images with different styles using different prompts while preserving the input structure.

```python
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-ControlNet")],
    lazy_loading=True,
)
```

```python
dataset_snapshot_download(
    "DiffSynth-Studio/examples_in_diffsynth",
    allow_file_pattern=["templates/*"],
    local_dir="data/examples",
)
image = template(
    pipe,
    prompt="A cat is sitting on a stone, bathed in bright sunshine.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "image": Image.open("data/examples/templates/image_depth.jpg"),
        "prompt": "A cat is sitting on a stone, bathed in bright sunshine.",
    }],
    negative_template_inputs = [{
        "image": Image.open("data/examples/templates/image_depth.jpg"),
        "prompt": "",
    }],
)
image.save("image_ControlNet_sunshine.jpg")
image = template(
    pipe,
    prompt="A cat is sitting on a stone, surrounded by colorful magical particles.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "image": Image.open("data/examples/templates/image_depth.jpg"),
        "prompt": "A cat is sitting on a stone, surrounded by colorful magical particles.",
    }],
    negative_template_inputs = [{
        "image": Image.open("data/examples/templates/image_depth.jpg"),
        "prompt": "",
    }],
)
image.save("image_ControlNet_magic.jpg")
```

```python
show_images([
    Image.open("data/examples/templates/image_depth.jpg"),
    Image.open("image_ControlNet_sunshine.jpg"),
    Image.open("image_ControlNet_magic.jpg"),
], resolution=256)
```
![Image](https://github.com/user-attachments/assets/048ee1d4-6f84-4edc-beb7-49bb5ec2d53d)

## Attribute Value Control

[AttriCtrl](https://arxiv.org/abs/2508.02151) is a type of controllable generation model capable of injecting continuous value attributes as control conditions into the generation process.

Run the following code to load [DiffSynth-Studio/Template-KleinBase4B-SoftRGB](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-SoftRGB) and precisely control the overall color tone of the image by inputting R/G/B values.

```python
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-SoftRGB")],
    lazy_loading=True,
)
```

```python
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{"R": 128/255, "G": 128/255, "B": 128/255}],
)
image.save("image_rgb_normal.jpg")
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{"R": 208/255, "G": 185/255, "B": 138/255}],
)
image.save("image_rgb_warm.jpg")
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{"R": 94/255, "G": 163/255, "B": 174/255}],
)
image.save("image_rgb_cold.jpg")
```

```python
show_images([
    Image.open("image_rgb_normal.jpg"),
    Image.open("image_rgb_warm.jpg"),
    Image.open("image_rgb_cold.jpg"),
], resolution=256)
```
![Image](https://github.com/user-attachments/assets/025ce94d-fe43-4166-8967-2acfbc76ada3)

## Image Editing

Image editing models are a type of **highly versatile** controllable generation model: given an original image and an editing instruction, the model can make partial or overall modifications to the original image.

Run the following code to load [DiffSynth-Studio/Template-KleinBase4B-Edit](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Edit). This model uses **KV-Cache** to reuse the attention key-value pairs of the input image, enabling fast editing with quick inference speed.

```python
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-Edit")],
    lazy_loading=True,
)
```

```python
dataset_snapshot_download(
    "DiffSynth-Studio/examples_in_diffsynth",
    allow_file_pattern=["templates/*"],
    local_dir="data/examples",
)
image = template(
    pipe,
    prompt="Put a hat on this cat.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "image": Image.open("data/examples/templates/image_reference.jpg"),
        "prompt": "Put a hat on this cat.",
    }],
    negative_template_inputs = [{
        "image": Image.open("data/examples/templates/image_reference.jpg"),
        "prompt": "",
    }],
)
image.save("image_Edit_hat.jpg")
image = template(
    pipe,
    prompt="Make the cat turn its head to look to the right.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "image": Image.open("data/examples/templates/image_reference.jpg"),
        "prompt": "Make the cat turn its head to look to the right.",
    }],
    negative_template_inputs = [{
        "image": Image.open("data/examples/templates/image_reference.jpg"),
        "prompt": "",
    }],
)
image.save("image_Edit_head.jpg")
```

```python
show_images([
    Image.open("data/examples/templates/image_reference.jpg"),
    Image.open("image_Edit_hat.jpg"),
    Image.open("image_Edit_head.jpg"),
], resolution=256)
```
![Image](https://github.com/user-attachments/assets/5cc0d8c3-6b4e-4fec-a36f-e58b6b00fe1a)

## Style Control

The most straightforward way to achieve image style control is to train a style [LoRA](https://arxiv.org/abs/2106.09685) — however, each style requires separate training, which is costly. To address this, we trained a special [Image-to-LoRA](https://arxiv.org/abs/2606.13809) model that can **generate LoRA weights on-demand from input reference images**, eliminating the traditional style training process.

Run the following code to load [DiffSynth-Studio/KleinBase4B-i2L-v2](https://www.modelscope.cn/models/DiffSynth-Studio/KleinBase4B-i2L-v2) and dynamically generate LoRA from reference images to control the image style.

```python
from modelscope import snapshot_download

template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/KleinBase4B-i2L-v2")],
    lazy_loading=True,
)
```

```python
snapshot_download("DiffSynth-Studio/KleinBase4B-i2L-v2", allow_file_pattern="assets/*", local_dir="data")
images = [Image.open(f"data/assets/image_1_{i}.jpg") for i in range(4)]
image = template(
    pipe,
    prompt="A cat is sitting on a stone",
    seed=42, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{"image": images}],
    negative_template_inputs = [{"image": [Image.fromarray(np.zeros_like(np.array(i)) + 128) for i in images]}],
)
image.save("image_KleinBase4B-i2L-v2_1.jpg")
images = [Image.open(f"data/assets/image_3_{i}.jpg") for i in range(4)]
image = template(
    pipe,
    prompt="A cat is sitting on a stone",
    seed=42, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{"image": images}],
    negative_template_inputs = [{"image": [Image.fromarray(np.zeros_like(np.array(i)) + 128) for i in images]}],
)
image.save("image_KleinBase4B-i2L-v2_2.jpg")
```

```python
show_images([
    Image.open("data/assets/image_1_2.jpg"),
    Image.open("image_KleinBase4B-i2L-v2_1.jpg"),
    Image.open("data/assets/image_3_0.jpg"),
    Image.open("image_KleinBase4B-i2L-v2_2.jpg"),
], resolution=256)
```
![Image](https://github.com/user-attachments/assets/2eb8abf6-4b7c-4f59-9f7e-69b4c3d3ce3a)

## Training Controllable Generation Models

**The Diffusion-Templates framework allows developers to train controllable generation models of any structure** — as long as you provide the model definition, data processing logic, and dataset, you can integrate into a unified training workflow. Below, we train a **brightness control model** from scratch, allowing images to be generated with specified brightness values.

Step 1: Write the model structure code (including the numerical encoder, KV-Cache generation backbone, and data annotator):

```python
code = """
import torch, math, os
from PIL import Image
import numpy as np


class SingleValueEncoder(torch.nn.Module):
    def __init__(self, dim_in=256, dim_out=4096, length=32):
        super().__init__()
        self.length = length
        self.prefer_value_embedder = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_out), torch.nn.SiLU(), torch.nn.Linear(dim_out, dim_out))
        self.positional_embedding = torch.nn.Parameter(torch.randn(self.length, dim_out))

    def get_timestep_embedding(self, timesteps, embedding_dim, max_period=10000):
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb

    def forward(self, value, dtype):
        emb = self.get_timestep_embedding(value * 1000, 256).to(dtype)
        emb = self.prefer_value_embedder(emb).squeeze(0)
        base_embeddings = emb.expand(self.length, -1)
        positional_embedding = self.positional_embedding.to(dtype=base_embeddings.dtype, device=base_embeddings.device)
        learned_embeddings = base_embeddings + positional_embedding
        return learned_embeddings


# Backbone model structure (converts input values into KV-Cache vectors)
class ValueFormatModel(torch.nn.Module):
    def __init__(self, num_double_blocks=5, num_single_blocks=20, dim=3072, num_heads=24, length=512):
        super().__init__()
        self.block_names = [f"double_{i}" for i in range(num_double_blocks)] + [f"single_{i}" for i in range(num_single_blocks)]
        self.proj_k = torch.nn.ModuleDict({block_name: SingleValueEncoder(dim_out=dim, length=length) for block_name in self.block_names})
        self.proj_v = torch.nn.ModuleDict({block_name: SingleValueEncoder(dim_out=dim, length=length) for block_name in self.block_names})
        self.num_heads = num_heads
        self.length = length

    @torch.no_grad()
    def process_inputs(self, pipe, scale, **kwargs):
        return {"value": torch.Tensor([scale]).to(dtype=pipe.torch_dtype, device=pipe.device)}

    def forward(self, value, **kwargs):
        kv_cache = {}
        for block_name in self.block_names:
            k = self.proj_k[block_name](value, value.dtype)
            k = k.view(1, self.length, self.num_heads, -1)
            v = self.proj_v[block_name](value, value.dtype)
            v = v.view(1, self.length, self.num_heads, -1)
            kv_cache[block_name] = (k, v)
        return {"kv_cache": kv_cache}


# Converts image data to model input (calculates brightness from RGB values in the image)
class DataAnnotator(torch.nn.Module):
    def __init__(self):
        pass

    def __call__(self, image, **kwargs):
        image = Image.open(image)
        image = np.array(image)
        return {"scale": image.astype(np.float32).mean() / 255}


TEMPLATE_MODEL = ValueFormatModel
TEMPLATE_MODEL_PATH = "model.safetensors" if "model.safetensors" in os.listdir(os.path.dirname(__file__)) else None
TEMPLATE_DATA_PROCESSOR = DataAnnotator
"""

import os

os.makedirs("models/template_brightness", exist_ok=True)
with open("models/template_brightness/model.py", "w", encoding="utf-8") as f:
    f.write(code.strip())
```

Step 2: Download and preprocess the dataset, while generating the metadata required for training:

```python
import json, os
from modelscope import dataset_snapshot_download

# Download dataset
dataset_snapshot_download(
    "DiffSynth-Studio/ImagePulseV2-TextImage",
    local_dir="data/ImagePulseV2-TextImage",
    allow_file_pattern="data/1770381050168240056.tar.gz"
)

# Extract dataset
os.makedirs("data/dataset", exist_ok=True)
os.system("tar zxvf data/ImagePulseV2-TextImage/data/1770381050168240056.tar.gz -C data/dataset")

# Generate dataset metadata
dataset_path = "data/dataset/1770381050168240056"
metadata = []
for file_name in os.listdir(dataset_path):
    if file_name.endswith(".json"):
        with open(os.path.join(dataset_path, file_name), "r") as f:
            data = json.load(f)
        data["template_inputs"] = {"image": os.path.join(dataset_path, data["image"])}
        metadata.append(data)
with open("data/dataset/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)
```

Step 3: Start training:

```python
import os

# Training script
code = """
import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Flux2ImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        template_model_id_or_path=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        enable_lora_hot_loading=False,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = self.parse_path_or_model_id(tokenizer_path, default_value=ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="tokenizer/"))
        self.pipe = Flux2ImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config)
        self.pipe = self.load_training_template_model(self.pipe, template_model_id_or_path, use_gradient_checkpointing, use_gradient_checkpointing_offload)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model, remove_unnecessary_params=True)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        if enable_lora_hot_loading: self.pipe.dit = self.pipe.enable_lora_hot_loading(self.pipe.dit)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }

    def get_pipeline_inputs(self, data):
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
            "embedded_guidance": 1.0,
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def flux2_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = flux2_parser()
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )
    model = Flux2ImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        template_model_id_or_path=args.template_model_id_or_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_lora_hot_loading=args.enable_lora_hot_loading,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
""".strip()
with open("train.py", "w", encoding="utf-8") as f:
    f.write(code)

# Start training task
cmd = """
accelerate launch train.py \
  --dataset_base_path data/dataset/1770381050168240056 \
  --dataset_metadata_path data/dataset/metadata.json \
  --extra_inputs "template_inputs" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-4B:vae/diffusion_pytorch_model.safetensors" \
  --template_model_id_or_path "DiffSynth-Studio/Template-KleinBase4B-Brightness:" \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-4B:tokenizer/" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.template_model." \
  --output_path "models/template_brightness_training" \
  --trainable_models "template_model" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --fp8_models "black-forest-labs/FLUX.2-klein-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors"
"""
os.system(cmd)
```

After training is complete, package the obtained weights together with the model definition written earlier into the `models/template_brightness` directory to form a complete Template model:

```python
import shutil

shutil.copy(
    "models/template_brightness_training/epoch-0.safetensors",
    "models/template_brightness/model.safetensors",
)
```

Load the trained model and generate images with different brightness by passing different `scale` values:

```python
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig("models/template_brightness")],
    lazy_loading=True,
)
```

```python
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{"scale": 0.7}],
    negative_template_inputs = [{"scale": 0.5}]
)
image.save("image_Brightness_light.jpg")
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{"scale": 0.3}],
    negative_template_inputs = [{"scale": 0.5}]
)
image.save("image_Brightness_dark.jpg")
```

```python
show_images([
    Image.open("image_Brightness_light.jpg"),
    Image.open("image_Brightness_dark.jpg"),
], resolution=256)
```
![Image](https://github.com/user-attachments/assets/ef15b73b-3c7a-4bb8-9b8e-1c4c37d9f3a1)
