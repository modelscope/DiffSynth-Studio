# Environment Variables

`DiffSynth-Studio` can control some settings through environment variables.

In `Python` code, you can set environment variables using `os.environ`. Please note that environment variables must be set before `import diffsynth`.

```python
import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "./path_to_my_models"
import diffsynth
```

On Linux operating systems, you can also temporarily set environment variables from the command line:

```shell
DIFFSYNTH_MODEL_BASE_PATH="./path_to_my_models" python xxx.py
```

Below are the environment variables supported by `DiffSynth-Studio`.

## `DIFFSYNTH_SKIP_DOWNLOAD`

Whether to skip model downloads. Can be set to `True`, `true`, `False`, `false`. If `skip_download` is not set in `ModelConfig`, this environment variable will determine whether to skip model downloads.

## `DIFFSYNTH_MODEL_BASE_PATH`

Model download root directory. Can be set to any local path. If `local_model_path` is not set in `ModelConfig`, model files will be downloaded to the path pointed to by this environment variable. If neither is set, model files will be downloaded to `./models`.

## `DIFFSYNTH_ATTENTION_IMPLEMENTATION`

Attention mechanism implementation method. Can be set to `flash_attention_3`, `flash_attention_2`, `sage_attention`, `xformers`, or `torch`. See [`./core/attention.md`](/docs/en/API_Reference/core/attention.md) for details.

## `DIFFSYNTH_DISK_MAP_BUFFER_SIZE`

Buffer size in disk mapping. Default is 1B (1000000000). Larger values occupy more memory but result in faster speeds.

## `DIFFSYNTH_DOWNLOAD_SOURCE`

Remote model download source. Can be set to `modelscope` or `huggingface` to control the source of model downloads. Default value is `modelscope`.