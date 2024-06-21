# ExVideo

ExVideo is a post-tuning technique aimed at enhancing the capability of video generation models. We have extended Stable Video Diffusion to achieve the generation of long videos up to 128 frames.

* [Project Page](https://ecnu-cilab.github.io/ExVideoProjectPage/)
* [Technical report](https://arxiv.org/abs/2406.14130)
* Extended models
    * [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1)
    * [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1)

## Example: Text-to-video via extended Stable Video Diffusion

Generate a video using a text-to-image model and our image-to-video model. See [ExVideo_svd_test.py](./ExVideo_svd_test.py).

https://github.com/modelscope/DiffSynth-Studio/assets/35051019/d97f6aa9-8064-4b5b-9d49-ed6001bb9acc

## Train

* Step 1: Install additional packages

```
pip install lightning deepspeed
```

* Step 2: Download base model (from [HuggingFace](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors) or [ModelScope](https://www.modelscope.cn/api/v1/models/AI-ModelScope/stable-video-diffusion-img2vid-xt/repo?Revision=master&FilePath=svd_xt.safetensors)) to `models/stable_video_diffusion/svd_xt.safetensors`.

* Step 3: Prepare datasets

```
path/to/your/dataset
├── metadata.json
└── videos
    ├── video_1.mp4
    ├── video_2.mp4
    └── video_3.mp4
```

where the `metadata.json` is

```
[
    {
        "path": "videos/video_1.mp4"
    },
    {
        "path": "videos/video_2.mp4"
    },
    {
        "path": "videos/video_3.mp4"
    }
]
```

* Step 4: Run

```
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u ExVideo_svd_train.py \
  --pretrained_path "models/stable_video_diffusion/svd_xt.safetensors" \
  --dataset_path "path/to/your/dataset" \
  --output_path "path/to/save/models" \
  --steps_per_epoch 8000 \
  --num_frames 128 \
  --height 512 \
  --width 512 \
  --dataloader_num_workers 2 \
  --learning_rate 1e-5 \
  --max_epochs 100
```

* Step 5: Post-process checkpoints

Calculate Exponential Moving Average (EMA) and package it using `safetensors`.

```
python ExVideo_ema.py --output_path "path/to/save/models/lightning_logs/version_xx" --gamma 0.9
```

* Step 6: Enjoy your model

The EMA model is at `path/to/save/models/lightning_logs/version_xx/checkpoints/epoch=xx-step=yyy-ema.safetensors`. Load it in [ExVideo_svd_test.py](./ExVideo_svd_test.py) and then enjoy your model.
