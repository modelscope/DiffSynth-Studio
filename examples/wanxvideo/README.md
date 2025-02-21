# WanX-Video

## Inference

WanX-Video-1.3B supports text-to-video and video-to-video. See [`./wanx_text_to_video.py`](./wanx_text_to_video.py).

TODO: add examples here after updating ckpts.

## Train

We support WanX-Video LoRA training. Here is a tutorial.

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
CUDA_VISIBLE_DEVICES="0" python examples/wanxvideo/train_wanx_t2v.py \
  --task data_process \
  --dataset_path data/example_dataset \
  --output_path ./models \
  --text_encoder_path "models/wanx/wanx2.1_t2v/cache/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "models/wanx/wanx2.1_t2v/cache/vae.pth" \
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

```shell
CUDA_VISIBLE_DEVICES="0" python examples/wanxvideo/train_wanx_t2v.py \
  --task train \
  --dataset_path data/example_dataset \
  --output_path ./models \
  --dit_path "models/wanx/wanx2.1_t2v/cache/wanx_t2v_1.3b.pth" \
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

```python
import torch
from diffsynth import ModelManager, WanxVideoPipeline, save_video, VideoData


# TODO: download models here.
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models([
    "models/wanx/wanx2.1_t2v/cache/wanx_t2v_1.3b.pth",
    "models/wanx/wanx2.1_t2v/cache/models_t5_umt5-xxl-enc-bf16.pth",
    "models/wanx/wanx2.1_t2v/cache/vae.pth",
])
model_manager.load_lora("models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)

pipe = WanxVideoPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# Text-to-video
video = pipe(
    prompt="...",
    negative_prompt="...",
    num_inference_steps=50,
    seed=0, tiled=True
)
save_video(video, "video_with_lora.mp4", fps=30, quality=5)
```
