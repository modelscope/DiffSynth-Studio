# Diffutoon-2

基于 MiniMax-H3 的真实视频转动漫（双 ControlNet：线稿 + 颜色）。

## 权重

[DiffSynth-Studio/Diffutoon-2](https://modelscope.cn/models/DiffSynth-Studio/Diffutoon-2)

- `line-art_module.safetensors`
- `color_module.safetensors`

基座：[MiniMax/MiniMax-H3](https://modelscope.cn/models/MiniMax/MiniMax-H3)（FL2VA）。Turbo：[lightx2v/Minimax-h3-Turbo](https://modelscope.cn/models/lightx2v/Minimax-h3-Turbo)。

## 示例

```bash
CUDA_VISIBLE_DEVICES=0 python examples/minimax_h3/model_inference/MiniMax-H3-Diffutoon.py
```

默认：模糊 σ=15，控制强度 0.4/0.4，4 步 turbo，`max_pixels=1032192`；长视频 328 帧窗口，用 `pin_latents` 钉住上一窗末尾 5 帧对应的 2 个干净 latent。

## 预处理

```bash
python -m diffsynth.utils.controlnet.video_lineart input.mp4 lineart.mp4
python -m diffsynth.utils.controlnet.video_blur input.mp4 blur.mp4 15
python -m diffsynth.utils.controlnet.video_caption input.mp4 captions.json --g0 0 --window 328 --context 5
```
