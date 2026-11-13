# Installing Dependencies

Install from source (recommended):

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

Install from PyPI (there may be delays in version updates; for latest features, install from source):

```
pip install diffsynth
```

To keep the framework lightweight, the base installation only installs the necessary dependencies. We provide some additional installation options:

* `[audio]`: Support for audio models, e.g., ACE-Step, MiniMax-Music3, etc.
* `[quant]`: For parameter quantization, enabling precisions such as NF4, INT8, NVFP4.
* `[training]`: For distributed large-scale pretraining.
* `[logger]`: To enable training loggers such as TensorBoard, SwanLab, etc.
* `[npu]`: For Ascend NPU devices with x86 architecture.
* `[npu_aarch64]`: For Ascend NPU devices with aarch64/ARM architecture.
* Dependencies of specific models
   * `[controlnet]`: For the ControlNet annotator detectors (Canny, depth, OpenPose, etc.) used by the ControlNet examples under `examples/flux` and `examples/qwen_image`.
   * `[infiniteyou]`: https://arxiv.org/abs/2503.16418
   * `[ses]`: https://arxiv.org/abs/2602.03208
   * `[nexusgen]`: https://arxiv.org/pdf/2504.21356
* `[all]`: Includes all dependencies except the "dependencies of specific models" above.

You can install multiple sets of dependencies with `pip install -e ".[audio,quant]"` or `pip install diffsynth[audio,quant]`.

## GPU/NPU Support

### NVIDIA GPU

Install as described above.

### AMD GPU

You need to install the `torch` package with ROCm support. Taking ROCm 6.4 (as of the article update date: December 15, 2025) on Linux as an example, run the following command:

```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

### Apple Silicon

On Apple Silicon devices, no changes to the installation steps are needed. However, since VRAM and memory are unified, replace all `"cuda"` in the code with `"mps"` or `"cpu"`.

### Moore Threads MUSA GPU

Follow the [torch_musa](https://github.com/MooreThreads/torch_musa) instructions to prepare a matching Moore Threads GPU driver, MUSA SDK, and MUSA-enabled PyTorch, or use a Moore Threads container image with these components preinstalled. Keep the PyTorch build matched to `torch_musa`; do not replace it with a standard CUDA build.

Install the [torchada](https://github.com/MooreThreads/torchada) adapter and DiffSynth-Studio from the official repository:

```shell
pip install "torchada>=0.1.106" --index-url https://pypi.org/simple
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

#### Optional: MUSA FlashAttention-3 Acceleration

FlashAttention-3 is not required by `torchada` itself. If using MUSA FlashAttention-3, upgrade the interface package and its companion MATE and MATE-MUBIN operator packages together:

```shell
pip install -U "mate[all]>=0.2.8.post1" "mate-mubin>=0.2.8.post1" \
    "flash_attn_3>=0.2.8.post1" \
    --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
    --extra-index-url https://pypi.org/simple
```

The tested combination is `flash_attn_3==0.2.8.post1+musa`, `mate==0.2.8.post1`, and `mate-mubin==0.2.8.post1`. Older MATE releases may have attention forward/backward or JIT compilation issues; these minimum versions do not imply that every subsequent release has been tested. If other operator packages in the base image pin an older MATE version, inspect the upgrade logs and `pip check` output and resolve conflicts according to those packages' compatibility requirements. Installing these packages does not guarantee that the current framework detects and selects FA3; check the registered implementations with `diffsynth.core.attention.list_attention_implementations()`.

For automatic import behavior and device-detection settings required before launch, see [MUSA GPU Support](GPU_support.md#moore-threads-musa-gpu). Install model-specific dependencies as described in the corresponding model documentation.

### Ascend NPU

1. Install [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit) through official documentation.

2. Install from source
   ```shell
   git clone https://github.com/modelscope/DiffSynth-Studio.git
   cd DiffSynth-Studio
   # aarch64/ARM
   pip install -e .[npu_aarch64] 
   # x86
   pip install -e .[npu] --extra-index-url "https://download.pytorch.org/whl/cpu"
   ```
When using Ascend NPU, please replace `"cuda"` with `"npu"` in your Python code. For details, see [NPU Support](../Pipeline_Usage/GPU_support.md#ascend-npu).

## Other Installation Issues

If you encounter issues during installation, they may be caused by upstream dependencies. Please refer to the documentation for these packages:

* [torch](https://pytorch.org/get-started/locally/)
* [Ascend/pytorch](https://github.com/Ascend/pytorch)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
