# LTX-2.5

DiffSynth-Studio 通过 `LTX25AudioVideoPipeline` 提供可移植的 LTX-2.5
音视频联合推理。该实现从本地加载官方 BF16 分组件权重，运行时不依赖
`ltx_core`、NATTEN、Triton 或 ltx-kernels。

> LTX-2.5 权重受门控保护。运行示例前，请先在 Lightricks 页面申请访问权限，
> 并将权重放到本地模型目录。

## 已实现的组件

- LTX-2.5 22B Distilled 和 Dev DiT
- LTX 微调 Gemma4 12B 编码器、内嵌 tokenizer 资产和音视频双 connector
- Duration Head 与因果帧网格的自动帧数预测
- DiffVAE 视频编码器和纯 PyTorch eager diffusion 解码器
- 音频 VAE、48 kHz BWE vocoder、空间 x2 latent upsampler 及时域 x2
  upsampler 注册
- Dev 第二阶段 distilled-LoRA 加载

Eager DiffVAE 解码器使用 tiled scaled-dot-product attention 实现可移植的
邻域注意力后备路径。固定输入下，其确定性解码阶段和一个 x0 diffusion step
与上游 eager 实现数值一致。

## 推理模式

公开的 `LTX25AudioVideoPipeline` API 与现有 LTX-2.3 API 对齐。

| 模式 | Pipeline 参数 | 状态 |
|---|---|---|
| Distilled 两阶段 T2AV | `use_distilled_pipeline=True`，`use_two_stage_pipeline=True` | 支持 |
| Distilled 两阶段 I2AV | `input_images`，`input_images_indexes` | 支持 |
| Dev 单阶段 T2AV | `use_distilled_pipeline=False`，`use_two_stage_pipeline=False` | 支持 |
| Dev 单阶段 I2AV | `input_images`，`use_two_stage_pipeline=False` | 支持 |
| Dev 两阶段 T2AV/I2AV | `stage2_lora_config`，`use_two_stage_pipeline=True` | 支持 |
| A2V | `retake_audio`，`audio_sample_rate`，可选 `retake_audio_regions` | 支持 |
| Video/audio Retake | `retake_video`，`retake_video_regions`，`retake_audio_regions` | 支持 |
| Keyframe interpolation | 多个 `input_images` 和 `input_images_indexes` | 支持 |
| Pixel Spatial Upscaler IC-LoRA | `in_context_videos`，`in_context_downsample_factor=2` | 支持 |

Dev 两阶段推理需要通过 `stage2_lora_config` 提供发布的第二阶段 distilled-LoRA。
Distilled 推理必须使用两阶段；Dev 同时支持不加载第二阶段 LoRA 的单阶段路径。

Pixel Spatial Upscaler 需要官方的 LTX-2.5 Pixel IC-LoRA。通过
`pipe.load_lora(pipe.dit, ModelConfig(path=...))` 加载它，将参考视频传给
`in_context_videos`，并设置 `clear_lora_before_state_two=True`。参考视频的
宽高应为最终输出的四分之一：第一阶段为半分辨率，adapter 的
`reference_downscale_factor` 为 2。不要将 LTX-2.3 adapter 加载到
LTX-2.5 DiT 中。

## 几何与显存要求

- `num_frames % 8 == 1`
- 单阶段的高度、宽度必须是 32 的倍数。
- 两阶段的高度、宽度必须是 64 的倍数。
- 低显存示例使用 BF16 计算、FP8 CPU 权重卸载，并为 DiT、Gemma4 编码器、
  text connector 和 DiffVAE decoder 启用细粒度显存管理。
- `LTX25_VRAM_LIMIT_GB` 控制保留在 GPU 上的预加载模型层预算，默认值为 16；
  它不是端到端显存硬上限。
- 在 `LTX25_VRAM_LIMIT_GB=16` 下，960×576×121 distilled T2AV 实测 PyTorch
  allocated 峰值为 31.8 GiB、reserved 峰值为 44.4 GiB。48 GiB 仅是该分辨率
  的实测下界，不代表已在 48 GiB 显存卡上验证，并应为驱动预留余量。
- 当前 `tiled=True` 不会切分 portable DiffVAE 的完整 decode volume。decoder
  决定了目前的全分辨率峰值；要支持更低显存卡，需要实现 tiled decoder。

运行低显存示例：

```bash
python examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-DistilledPipeline.py
python examples/ltx2/model_inference_low_vram/LTX-2.5-Keyframe-Interpolation.py
python examples/ltx2/model_inference_low_vram/LTX-2.5-IC-LoRA-Pixel-Spatial-Upscaler.py
```

分组件权重默认位于 `models/Lightricks/LTX-2.5`。Pixel 示例还需要单独门控的
adapter，默认位于
`models/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`。显存充足时可以
通过 `LTX25_VRAM_LIMIT_GB=24` 增大预加载层预算以改善速度，但它不会降低 decoder
峰值。如本地模型目录不同，请修改示例中的路径。

## 范围

该接入仅覆盖推理。训练、LTX-2.5 专有 DFR、Dub-It 和 HDR/EXR
pipeline 不在本次范围内。
