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

EntroPack is an optional dependency. Before running quantization or low-VRAM examples that use `entropack_*` methods, install the matching backend as described in the [quantization guide](Quantization.md). For example, use `python -m pip install "entropack[cuda13]>=0.1.7"` for CUDA 13, or replace `cuda13` with `cuda12` for CUDA 12. This also applies to NVIDIA GPU users.

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

### Moore Threads MUSA

Install matching `torch`, `torch_musa`, and `torchvision` builds from a MUSA image or vendor distribution first, then run `python -m pip install -e .` with that environment's Python. The base installation reuses installed packages that satisfy its requirements; do not add `--upgrade`.

`entropack>=0.1.7` requires `torch>=2.10`, so it is only included in `[quant]` / `[all]`. Do not install those extras in MUSA environments using torch 2.7.1 / 2.9.1. Quantization and low-VRAM examples using EntroPack still require a separately installed compatible backend; the base installation does not provide those features.

Reuse the audio and training packages validated in your image. If you need `[audio]` / `[training]`, protect the installed framework versions with an environment-specific constraints file and inspect the plan with `python -m pip install --dry-run -c <constraints.txt> -e ".[audio,training]"` first. In particular, audio dependencies must not replace the MUSA torch build.

## Other Installation Issues

If you encounter issues during installation, they may be caused by upstream dependencies. Please refer to the documentation for these packages:

* [torch](https://pytorch.org/get-started/locally/)
* [Ascend/pytorch](https://github.com/Ascend/pytorch)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
