# Wan-Video

Wan-Video is a collection of video synthesis models open-sourced by Alibaba.

## Inference

### Wan-Video-1.3B-T2V

Wan-Video-1.3B-T2V supports text-to-video and video-to-video. See [`./wan_1.3b_text_to_video.py`](./wan_1.3b_text_to_video.py).

Required VRAM: 6G

https://github.com/user-attachments/assets/124397be-cd6a-4f29-a87c-e4c695aaabb8

Put sunglasses on the dog.

https://github.com/user-attachments/assets/272808d7-fbeb-4747-a6df-14a0860c75fb

### Wan-Video-14B-T2V

Wan-Video-14B-T2V is an enhanced version of Wan-Video-1.3B-T2V, offering greater size and power. To utilize this model, you need additional VRAM. We recommend that users adjust the `torch_dtype` and `num_persistent_param_in_dit` settings to find an optimal balance between speed and VRAM requirements. See [`./wan_14b_text_to_video.py`](./wan_14b_text_to_video.py).

We present a detailed table here. The model is tested on a single A100.

|`torch_dtype`|`num_persistent_param_in_dit`|Speed|Required VRAM|Default Setting|
|-|-|-|-|-|
|torch.bfloat16|None (unlimited)|18.5s/it|40G||
|torch.bfloat16|7*10**9 (7B)|20.8s/it|24G||
|torch.bfloat16|0|23.4s/it|10G||
|torch.float8_e4m3fn|None (unlimited)|18.3s/it|24G|yes|
|torch.float8_e4m3fn|0|24.0s/it|10G||

https://github.com/user-attachments/assets/3908bc64-d451-485a-8b61-28f6d32dd92f

### Wan-Video-14B-I2V

Wan-Video-14B-I2V adds the functionality of image-to-video based on Wan-Video-14B-T2V. The model size remains the same, therefore the speed and VRAM requirements are also consistent. See [`./wan_14b_image_to_video.py`](./wan_14b_image_to_video.py).

![Image](https://github.com/user-attachments/assets/adf8047f-7943-4aaa-a555-2b32dc415f39)

https://github.com/user-attachments/assets/c0bdd5ca-292f-45ed-b9bc-afe193156e75

## Train

We support Wan-Video LoRA training and full training. Here is a tutorial.

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
    └── video_00002.mp4
```

`metadata.csv`:

```
file_name,text
video_00001.mp4,"video description"
video_00001.mp4,"video description"
```

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
  --lora_rank 4 \
  --lora_alpha 4 \
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

Step 4-1: I2V LoRA-training
```shell
# cache latents
CUDA_VISIBLE_DEVICES="0" python train_wan_i2v.py \
    --task data_process \
    --dataset_path data/fps24_V6 \
    --output_path ./output \
    --text_encoder_path "./models/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth" \
    --vae_path "./models/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth" \
    --image_encoder_path "./models/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --tiled \
    --num_frames 121 \
    --height 309 \
    --width 186
```

```shell
# run I2V training 
CUDA_VISIBLE_DEVICES="0" python train_wan_i2v.py \
    --task train \
    --train_architecture lora \
    --dataset_path data/kling_hips_fps24_V6  \
    --output_path ./output \
    --dit_path "[
            \"./models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors\",
            \"./models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors\",
            \"./models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors\",
            \"./models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors\",
            \"./models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors\",
            \"./models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors\",
            \"./models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors\"
        ]"  \
    --steps_per_epoch 500 \
    --max_epochs 10 \
    --learning_rate 1e-4 \
    --lora_rank 4 \
    --lora_alpha 4 \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --accumulate_grad_batches 1 \
    --use_gradient_checkpointing
```
 
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
