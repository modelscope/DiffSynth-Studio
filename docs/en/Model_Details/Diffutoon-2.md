# Diffutoon-2

Real-world video to Japanese anime-style video on MiniMax-H3 with dual ControlNets (line-art + color).

## Weights

[DiffSynth-Studio/Diffutoon-2](https://modelscope.cn/models/DiffSynth-Studio/Diffutoon-2)

- `line-art_module.safetensors`
- `color_module.safetensors`

Base: [MiniMax/MiniMax-H3](https://modelscope.cn/models/MiniMax/MiniMax-H3) (FL2VA). Turbo: [lightx2v/Minimax-h3-Turbo](https://modelscope.cn/models/lightx2v/Minimax-h3-Turbo).

## Example

```bash
CUDA_VISIBLE_DEVICES=0 python examples/minimax_h3/model_inference/MiniMax-H3-Diffutoon.py
```

Defaults: blur σ=15, control scales 0.4/0.4, 4-step turbo, `max_pixels=1032192`, long clips use 328-frame windows with a 5-frame (2-latent) pin via `pin_latents`.

## Preprocess

```bash
python -m diffsynth.utils.controlnet.video_lineart input.mp4 lineart.mp4
python -m diffsynth.utils.controlnet.video_blur input.mp4 blur.mp4 15
python -m diffsynth.utils.controlnet.video_caption input.mp4 captions.json --g0 0 --window 328 --context 5
```
