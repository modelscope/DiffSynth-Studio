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

## GPU/NPU Support

* **NVIDIA GPU**

Install as described above.

* **AMD GPU**

You need to install the `torch` package with ROCm support. Taking ROCm 6.4 (as of the article update date: December 15, 2025) on Linux as an example, run the following command:

```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

* **Ascend NPU**

1. Install [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit) through official documentation.

2. Install from source
   ```shell
   git clone https://github.com/modelscope/DiffSynth-Studio.git
   cd DiffSynth-Studio
   # aarch64/ARM
   pip install -e .[npu_aarch64] --extra-index-url "https://download.pytorch.org/whl/cpu"
   # x86
   pip install -e .[npu]

When using Ascend NPU, please replace `"cuda"` with `"npu"` in your Python code. For details, see [NPU Support](/docs/en/Pipeline_Usage/GPU_support.md#ascend-npu).

## Other Installation Issues

If you encounter issues during installation, they may be caused by upstream dependencies. Please refer to the documentation for these packages:

* [torch](https://pytorch.org/get-started/locally/)
* [Ascend/pytorch](https://github.com/Ascend/pytorch)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
