# 魔搭社区 AIGC 系列课程 - 可控生成技术

本实验以 **Diffusion-Templates** 为框架，系统介绍图像生成模型的多种可控生成技术，并演示如何自行训练一个可控生成模块。

相关资料：

* 开源代码：[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* 技术报告：[arXiv](https://arxiv.org/abs/2604.24351)
* 项目主页：[GitHub](https://modelscope.github.io/diffusion-templates-web/)
* 文档参考：[English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)、[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)
* 在线体验：[魔搭社区创空间](https://modelscope.cn/studios/DiffSynth-Studio/Diffusion-Templates)
* 模型集：[ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/KleinBase4B-Templates)、[ModelScope 国际站](https://modelscope.ai/collections/DiffSynth-Studio/KleinBase4B-Templates)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/kleinbase4b-templates)
* 数据集：[ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[ModelScope 国际站](https://modelscope.ai/collections/DiffSynth-Studio/ImagePulseV2)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/imagepulsev2)

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

## 图像结构控制

首先，加载基础模型 [black-forest-labs/FLUX.2-klein-base-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B)。这是一个参数量为 4B 的图像生成模型，本实验后续所有可控生成模块都会挂载到这个基础模型之上。

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

[ControlNet](https://arxiv.org/abs/2302.05543) 是最早的一批 Diffusion 可控生成技术，可用**深度图、边缘图、姿态图**等结构性条件对生成画面进行**逐像素级**的控制。

以 Template 格式加载 [DiffSynth-Studio/Template-KleinBase4B-ControlNet](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-ControlNet)，即可在保留输入结构的前提下，用不同的提示词生成不同风格的画面。

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

## 数值属性控制

[AttriCtrl](https://arxiv.org/abs/2508.02151) 是一类**数值型**可控生成模型，能够将连续的数值属性作为控制条件注入生成过程。

运行以下代码，加载 [DiffSynth-Studio/Template-KleinBase4B-SoftRGB](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-SoftRGB)，通过输入 R/G/B 数值精确控制画面的整体色调。

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

## 图像编辑

图像编辑模型是一类**通用性较强**的可控生成模型：给定一张原图和一段编辑指令，即可对原图进行局部或整体修改。

运行以下代码，加载 [DiffSynth-Studio/Template-KleinBase4B-Edit](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Edit)。该模型通过 **KV-Cache** 复用输入图像的注意力键值，从而快速完成编辑，推理速度较快。

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
![Image](https://github.com/user-attachments/assets/73140bc8-e510-4832-b3f3-40c00281e136)

## 风格控制

实现图像风格控制的最直接方式，是训练一个风格 [LoRA](https://arxiv.org/abs/2106.09685)——但每种风格都需要单独训练，成本较高。为此我们训练了一个特殊的 [Image-to-LoRA](https://arxiv.org/abs/2606.13809) 模型，它可以**根据输入的参考图像即时生成一份 LoRA 权重**，免去了传统的风格训练过程。

运行以下代码，加载 [DiffSynth-Studio/KleinBase4B-i2L-v2](https://www.modelscope.cn/models/DiffSynth-Studio/KleinBase4B-i2L-v2)，用参考图像动态生成 LoRA，从而控制画面风格。

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
![Image](https://github.com/user-attachments/assets/783748e7-90fc-494f-a939-70ea45e1486a)

## 训练可控生成模型

**Diffusion-Templates 框架允许开发者训练任意结构的可控生成模型**——只要给定模型定义、数据处理逻辑和数据集，即可接入统一的训练流程。下面我们从零训练一个**亮度控制模型**，让画面按指定的亮度数值生成。

第一步，编写模型结构代码（包含数值编码器、KV-Cache 生成主干和数据标注器）：

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


# 主干模型结构（将输入的数值转换为 KV-Cache 向量）
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


# 将图像数据转换为模型输入（根据图像中的 RGB 数值计算亮度）
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

第二步，下载并预处理数据集，同时生成训练所需的 metadata：

```python
import json, os
from modelscope import dataset_snapshot_download

# 下载数据集
dataset_snapshot_download(
    "DiffSynth-Studio/ImagePulseV2-TextImage",
    local_dir="data/ImagePulseV2-TextImage",
    allow_file_pattern="data/1770381050168240056.tar.gz"
)

# 解压数据集
os.makedirs("data/dataset", exist_ok=True)
os.system("tar zxvf data/ImagePulseV2-TextImage/data/1770381050168240056.tar.gz -C data/dataset")

# 生成数据集 metadata
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

第三步，启动训练：

```python
import os

# 训练脚本
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

# 启动训练任务
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

训练完成后，将得到的权重与前面写好的模型定义一起打包到 `models/template_brightness` 目录，形成一个完整的 Template 模型：

```python
import shutil

shutil.copy(
    "models/template_brightness_training/epoch-0.safetensors",
    "models/template_brightness/model.safetensors",
)
```

加载训练好的模型，通过传入不同的 `scale` 数值生成明暗不同的图像：

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
![Image](https://github.com/user-attachments/assets/f3b72cb5-4d7d-46ca-82c8-1ede4d71af1e)