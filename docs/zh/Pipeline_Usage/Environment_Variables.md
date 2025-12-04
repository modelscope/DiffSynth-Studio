# 环境变量

`DiffSynth-Studio` 可通过环境变量控制一些设置。

在 `Python` 代码中，可以使用 `os.environ` 设置环境变量。请注意，环境变量需在 `import diffsynth` 前设置。

```python
import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "./path_to_my_models"
import diffsynth
```

在 Linux 操作系统上，也可在命令行临时设置环境变量：

```shell
DIFFSYNTH_MODEL_BASE_PATH="./path_to_my_models" python xxx.py
```

以下是 `DiffSynth-Studio` 所支持的环境变量。

## `DIFFSYNTH_SKIP_DOWNLOAD`

是否跳过模型下载。可设置为 `True`、`true`、`False`、`false`，若 `ModelConfig` 中没有设置 `skip_download`，则会根据这一环境变量决定是否跳过模型下载。

## `DIFFSYNTH_MODEL_BASE_PATH`

模型下载根目录。可设置为任意本地路径，若 `ModelConfig` 中没有设置 `local_model_path`，则会将模型文件下载到这一环境变量指向的路径。若两者都未设置，则会将模型文件下载到 `./models`。

## `DIFFSYNTH_ATTENTION_IMPLEMENTATION`

注意力机制实现的方式，可以设置为 `flash_attention_3`、`flash_attention_2`、`sage_attention`、`xformers`、`torch`。详见 [`./core/attention.md`](/docs/zh/API_Reference/core/attention.md).

## `DIFFSYNTH_DISK_MAP_BUFFER_SIZE`

硬盘直连中的 Buffer 大小，默认是 1B（1000000000），数值越大，占用内存越大，速度越快。

## `DIFFSYNTH_DOWNLOAD_SOURCE`

远程模型下载源，可设置为 `modelscope` 或 `huggingface`，控制模型下载的来源，默认值为 `modelscope`。
