# Wan-Video

Wan-Video is a collection of video synthesis models open-sourced by Alibaba.

Before using this model, please install DiffSynth-Studio from **source code**.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

## Model Zoo

|Developer|Name|Link|Scripts|
|-|-|-|-|
|Wan Team|1.3B text-to-video|[Link](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)|[wan_1.3b_text_to_video.py](./wan_1.3b_text_to_video.py)|
|Wan Team|14B text-to-video|[Link](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)|[wan_14b_text_to_video.py](./wan_14b_text_to_video.py)|
|Wan Team|14B image-to-video 480P|[Link](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)|[wan_14b_image_to_video.py](./wan_14b_image_to_video.py)|
|Wan Team|14B image-to-video 720P|[Link](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)|[wan_14b_image_to_video.py](./wan_14b_image_to_video.py)|
|Wan Team|14B first-last-frame-to-video 720P|[Link](https://modelscope.cn/models/Wan-AI/Wan2.1-FLF2V-14B-720P)|[wan_14B_flf2v.py](./wan_14B_flf2v.py)|
|DiffSynth-Studio Team|1.3B aesthetics LoRA|[Link](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-aesthetics-v1)|Please see the [model card](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-aesthetics-v1).|
|DiffSynth-Studio Team|1.3B Highres-fix LoRA|[Link](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-highresfix-v1)|Please see the [model card](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-highresfix-v1).|
|DiffSynth-Studio Team|1.3B ExVideo LoRA|[Link](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-exvideo-v1)|Please see the [model card](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-exvideo-v1).|
|DiffSynth-Studio Team|1.3B Speed Control adapter|[Link](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1)|[wan_1.3b_motion_controller.py](./wan_1.3b_motion_controller.py)|
|PAI Team|1.3B InP|[Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP)|[wan_fun_InP.py](./wan_fun_InP.py)|
|PAI Team|14B InP|[Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP)|[wan_fun_InP.py](./wan_fun_InP.py)|
|PAI Team|1.3B Control|[Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control)|[wan_fun_control.py](./wan_fun_control.py)|
|PAI Team|14B Control|[Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-Control)|[wan_fun_control.py](./wan_fun_control.py)|
|IIC Team|1.3B VACE|[Link](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)|[wan_1.3b_vace.py](./wan_1.3b_vace.py)|

Base model features

||Text-to-video|Image-to-video|End frame|Control|Reference image|
|-|-|-|-|-|-|
|1.3B text-to-video|✅|||||
|14B text-to-video|✅|||||
|14B image-to-video 480P||✅||||
|14B image-to-video 720P||✅||||
|14B first-last-frame-to-video 720P||✅|✅|||
|1.3B InP||✅|✅|||
|14B InP||✅|✅|||
|1.3B Control||||✅||
|14B Control||||✅||
|1.3B VACE||||✅|✅|

Adapter model compatibility

||1.3B text-to-video|1.3B InP|1.3B VACE|
|-|-|-|-|
|1.3B aesthetics LoRA|✅||✅|
|1.3B Highres-fix LoRA|✅||✅|
|1.3B ExVideo LoRA|✅||✅|
|1.3B Speed Control adapter|✅|✅|✅|

## VRAM Usage

* Fine-grained offload: We recommend that users adjust the `num_persistent_param_in_dit` settings to find an optimal balance between speed and VRAM requirements. See [`./wan_14b_text_to_video.py`](./wan_14b_text_to_video.py).

* FP8 Quantization: You only need to adjust the `torch_dtype` in the `ModelManager` (not the pipeline!).

We present a detailed table here. The model (14B text-to-video) is tested on a single A100.

|`torch_dtype`|`num_persistent_param_in_dit`|Speed|Required VRAM|Default Setting|
|-|-|-|-|-|
|torch.bfloat16|None (unlimited)|18.5s/it|48G||
|torch.bfloat16|7*10**9 (7B)|20.8s/it|24G||
|torch.bfloat16|0|23.4s/it|10G||
|torch.float8_e4m3fn|None (unlimited)|18.3s/it|24G|yes|
|torch.float8_e4m3fn|0|24.0s/it|10G||

**We found that 14B image-to-video model is more sensitive to precision, so when the generated video content experiences issues such as artifacts, please switch to bfloat16 precision and use the `num_persistent_param_in_dit` parameter to control VRAM usage.**

## Efficient Attention Implementation

DiffSynth-Studio supports multiple Attention implementations. If you have installed any of the following Attention implementations, they will be enabled based on priority. However, we recommend to use the default torch SDPA.

* [Flash Attention 3](https://github.com/Dao-AILab/flash-attention)
* [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
* [Sage Attention](https://github.com/thu-ml/SageAttention)
* [torch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (default. `torch>=2.5.0` is recommended.)

## Acceleration

We support multiple acceleration solutions:
* [TeaCache](https://github.com/ali-vilab/TeaCache): See [wan_1.3b_text_to_video_accelerate.py](./wan_1.3b_text_to_video_accelerate.py).

* [Unified Sequence Parallel](https://github.com/xdit-project/xDiT): See [wan_14b_text_to_video_usp.py](./wan_14b_text_to_video_usp.py)

```bash
pip install xfuser>=0.4.3
torchrun --standalone --nproc_per_node=8 examples/wanvideo/wan_14b_text_to_video_usp.py
```

* Tensor Parallel: See [wan_14b_text_to_video_tensor_parallel.py](./wan_14b_text_to_video_tensor_parallel.py).

## Gallery

1.3B text-to-video.

https://github.com/user-attachments/assets/124397be-cd6a-4f29-a87c-e4c695aaabb8

Put sunglasses on the dog.

https://github.com/user-attachments/assets/272808d7-fbeb-4747-a6df-14a0860c75fb

14B text-to-video.

https://github.com/user-attachments/assets/3908bc64-d451-485a-8b61-28f6d32dd92f

14B image-to-video.

https://github.com/user-attachments/assets/c0bdd5ca-292f-45ed-b9bc-afe193156e75

14B first-last-frame-to-video

|First frame|Last frame|Video|
|-|-|-|
|![Image](https://github.com/user-attachments/assets/b0d8225b-aee0-4129-b8e5-58c8523221a6)|![Image](https://github.com/user-attachments/assets/2f0c9bc5-07e2-45fa-8320-53d63a4fd203)|https://github.com/user-attachments/assets/2a6a2681-622c-4512-b852-5f22e73830b1|

## Train

We support Wan-Video LoRA training and full training. Here is a tutorial. This is an experimental feature. Below is a video sample generated from the character Keqing LoRA:

https://github.com/user-attachments/assets/9bd8e30b-97e8-44f9-bb6f-da004ba376a9

Step 1: Install additional packages

```
pip install peft lightning pandas
```

Step 2: Prepare your dataset

You need to manage the training videos as follows:

```
data/example_dataset/
├── metadata.csv
└── train
    ├── video_00001.mp4
    └── image_00002.jpg
```

`metadata.csv`:

```
file_name,text
video_00001.mp4,"video description"
image_00002.jpg,"video description"
```

We support both images and videos. An image is treated as a single frame of video.

Step 3: Data process

```shell
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/train_wan_t2v.py \
  --task data_process \
  --dataset_path data/example_dataset \
  --output_path ./models \
  --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 81 \
  --height 480 \
  --width 832
```

After that, some cached files will be stored in the dataset folder.

```
data/example_dataset/
├── metadata.csv
└── train
    ├── video_00001.mp4
    ├── video_00001.mp4.tensors.pth
    ├── video_00002.mp4
    └── video_00002.mp4.tensors.pth
```

Step 4: Train

LoRA training:

```shell
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/train_wan_t2v.py \
  --task train \
  --train_architecture lora \
  --dataset_path data/example_dataset \
  --output_path ./models \
  --dit_path "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 500 \
  --max_epochs 10 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing
```

Full training:

```shell
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/train_wan_t2v.py \
  --task train \
  --train_architecture full \
  --dataset_path data/example_dataset \
  --output_path ./models \
  --dit_path "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 500 \
  --max_epochs 10 \
  --learning_rate 1e-4 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing
```

If you wish to train the 14B model, please separate the safetensor files with a comma. For example: `models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors`.

If you wish to train the image-to-video model, please add an extra parameter `--image_encoder_path "models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"`.

For LoRA training, the Wan-1.3B-T2V model requires 16G of VRAM for processing 81 frames at 480P, while the Wan-14B-T2V model requires 60G of VRAM for the same configuration. To further reduce VRAM requirements by 20%-30%, you can include the parameter `--use_gradient_checkpointing_offload`.

Step 5: Test

Test LoRA:

```python
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData


model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models([
    "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
])
model_manager.load_lora("models/lightning_logs/version_1/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

video = pipe(
    prompt="...",
    negative_prompt="...",
    num_inference_steps=50,
    seed=0, tiled=True
)
save_video(video, "video.mp4", fps=30, quality=5)
```

Test fine-tuned base model:

```python
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData


model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models([
    "models/lightning_logs/version_1/checkpoints/epoch=0-step=500.ckpt",
    "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
])
pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

video = pipe(
    prompt="...",
    negative_prompt="...",
    num_inference_steps=50,
    seed=0, tiled=True
)
save_video(video, "video.mp4", fps=30, quality=5)
```
