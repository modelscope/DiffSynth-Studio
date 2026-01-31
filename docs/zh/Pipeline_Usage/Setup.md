# 安装依赖

从源码安装（推荐）：

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

从 pypi 安装（存在版本更新延迟，如需使用最新功能，请从源码安装）

```
pip install diffsynth
```

## GPU/NPU 支持

* NVIDIA GPU

按照以上方式安装即可。

* AMD GPU

需安装支持 ROCm 的 `torch` 包，以 ROCm 6.4（本文更新于 2025 年 12 月 15 日）、Linux 系统为例，请运行以下命令

```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

* Ascend NPU

1. 通过官方文档安装[CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)

2. 从源码安装
   ```shell
   git clone https://github.com/modelscope/DiffSynth-Studio.git
   cd DiffSynth-Studio
   # aarch64/ARM
   pip install -e .[npu_aarch64] --extra-index-url "https://download.pytorch.org/whl/cpu"
   # x86
   pip install -e .[npu]

使用 Ascend NPU 时，请将 Python 代码中的 `"cuda"` 改为 `"npu"`，详见[NPU 支持](/docs/zh/Pipeline_Usage/GPU_support.md#ascend-npu)。

## 其他安装问题

如果在安装过程中遇到问题，可能是由上游依赖包导致的，请参考这些包的文档：

* [torch](https://pytorch.org/get-started/locally/)
* [Ascend/pytorch](https://github.com/Ascend/pytorch)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
