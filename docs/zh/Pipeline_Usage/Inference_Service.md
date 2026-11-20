# 部署推理服务

`services/` 目录是魔搭社区 AIGC 专区使用的模型推理服务代码。它把 DiffSynth-Studio 的 Pipeline 封装成统一的“模型服务”接口，面向 **H20 单 GPU 机器组成的集群**：每台机器只有一张 GPU，集群通过增加机器数量横向扩展，每台机器上运行一个或多个模型服务实例。

与 `examples/` 下面向单次运行的推理脚本不同，`services/` 中的每个类代表一个**常驻服务**：

- `load_models(device, deploy_mode)` 在服务启动时加载模型，并驻留在显存中；
- `infer(**kwargs)` 处理一次请求，返回图片、音频张量，或直接把视频写到 `output_path`；
- `example_inputs()` 返回与 `examples/` 中对应推理脚本一致的测试样例；
- `run_tests()` 对每个 `deploy_mode` 做完整的加载与推理基准测试，输出显存、内存和耗时表格。

## 服务清单

| 模块 | 服务类 | 模型 | 任务 | 输出 | deploy_mode |
|------|--------|------|------|------|-------------|
| `ace_step` | `AceStep15XLSFT` | ACE-Step 1.5 XL SFT | 文本生成音乐 | 音频张量 | 0 / 1 / 2 |
| `fibo` | `Fibo20Dev` | Fibo 2.0 dev | 文生图、图片编辑、蒙版编辑 | `PIL.Image` | 0 / 1 / 2 |
| `fibo` | `Fibo20EditTurbo` | Fibo 2.0 edit turbo | 图片编辑、蒙版编辑 | `PIL.Image` | 0 / 1 / 2 |
| `flux2` | `Flux2Klein4B` | FLUX.2-klein-4B | 文生图、图片编辑 | `PIL.Image` | 0 / 1 / 2 |
| `flux2` | `Flux2KleinBase4B` | FLUX.2-klein-base-4B | 文生图、图片编辑 | `PIL.Image` | 0 / 1 / 2 |
| `flux2` | `Flux2Klein9B` | FLUX.2-klein-9B | 文生图、图片编辑 | `PIL.Image` | 0 / 1 / 2 |
| `flux2` | `Flux2KleinBase9B` | FLUX.2-klein-base-9B | 文生图、图片编辑 | `PIL.Image` | 0 / 1 / 2 |
| `hidream_o1` | `HiDreamO1Image` | HiDream-O1-Image | 文生图、多图参考编辑 | `PIL.Image` | 0 / 1 / 2 |
| `krea2_turbo` | `Krea2Turbo` | Krea-2-Turbo | 文生图 | `PIL.Image` | 0 / 1 / 2 |
| `minimax_h3` | `MiniMaxH3FL2VA` | MiniMax-H3 FL2VA | 文生音视频、首尾帧生音视频 | 写入 mp4 | 0 / 1 |
| `minimax_h3` | `MiniMaxH3Ref2VA` | MiniMax-H3 Ref2VA | 参考图/视频/音频生音视频 | 写入 mp4 | 0 / 1 |
| `minimax_h3_image` | `MiniMaxH3Image` | MiniMax-H3-Image | 文生图、图片编辑 | `PIL.Image` | 0 |
| `minimax_music` | `MiniMaxMusic3` | MiniMax-Music3 | 文本生成音乐 | 音频张量 | 0 / 1 / 2 |
| `moss_image` | `MOSSImageSFT` | MOSS-Image SFT | 文生图 | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `QwenImage` | Qwen-Image | 文生图（可选 Context Control） | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `QwenImage2512` | Qwen-Image-2512 | 文生图 | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `QwenImageEdit` | Qwen-Image-Edit | 单图编辑 | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `QwenImageEdit2509` | Qwen-Image-Edit-2509 | 多图编辑 | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `QwenImageEdit2511` | Qwen-Image-Edit-2511 | 多图编辑 | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `FireRedImageEdit` | FireRed-Image-Edit-1.0 | 多图编辑 | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image_21` | `QwenImage21` | Qwen-Image-2.1 | 文生图（RGBA）、多图编辑 | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image_21` | `QwenImage21Turbo` | Qwen-Image-2.1-Turbo | 文生图（RGBA）、多图编辑 | `PIL.Image` | 0 / 1 / 2 |
| `z_image` | `ZImageTurbo` | Z-Image-Turbo | 文生图 | `PIL.Image` | 0 / 1 / 2 |
| `z_image` | `ZImage` | Z-Image | 文生图 | `PIL.Image` | 0 / 1 / 2 |

## 环境准备

推荐从已安装 PyTorch 的环境 clone，避免重复安装重型依赖：

```shell
conda create -n diffsynth_service --clone <your-pytorch-env> -y
conda activate diffsynth_service
cd DiffSynth-Studio
pip install -e .
```

DiffSynth-Studio 会自动从 ModelScope 下载模型。建议把模型统一放在共享存储上，并通过环境变量指定缓存目录：

```shell
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/diffsynth_models/
```

首次启动时各服务会自动下载所需权重。如果希望在部署前一次性预热模型缓存，可以运行：

```shell
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/diffsynth_models/
python data/diffsynth_service_tests/tools/download_models.py
```

如果使用 EntroPack 的量化权重（`deploy_mode` 1 / 2 以及 `MiniMaxH3Image`），需要安装 CuPy：

```shell
pip install cupy
```

`deploy_mode` 0 中的 NF4 权重（`MiniMaxH3FL2VA`、`MiniMaxH3Ref2VA`）还需要 `bitsandbytes`：

```shell
pip install bitsandbytes
```

## 快速开始

以 Qwen-Image 的文生图为例：

```python
import torch
from services.qwen_image import QwenImage

service = QwenImage()
service.load_models(device="cuda", deploy_mode=0)

image = service.infer(
    prompt="精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。",
    negative_prompt="",
    seed=0,
    cfg_scale=4,
    num_inference_steps=40,
    height=1472,
    width=1104,
)
image.save("output.png")
```

`load_models` 只需要在服务启动时调用一次。之后每个请求直接调用 `infer`，模型权重会常驻显存。

不同输出类型的处理方式：

```python
# 图片：infer 返回 PIL.Image
image.save("output.png")

# 音频：infer 返回 torch.Tensor，采样率从 pipeline 的声码器读取
from diffsynth.utils.data.audio import save_audio
from services.ace_step import AceStep15XLSFT
service = AceStep15XLSFT()
service.load_models(device="cuda", deploy_mode=1)
audio = service.infer(
    prompt="An upbeat acoustic pop song.",
    lyrics="[Verse]\nMorning light filtering through the pine",
    duration=30,
    vocal_language="en",
    keyscale="C major",
    bpm=96,
    timesignature="4",
    seed=7,
    cfg_scale=4,
    num_inference_steps=30,
)
save_audio(audio, service.pipe.vae.sampling_rate, "output.wav")

# 视频：infer 不返回值，直接写入 output_path 指定的 mp4（含音轨）
from services.minimax_h3 import MiniMaxH3FL2VA
service = MiniMaxH3FL2VA()
service.load_models(device="cuda", deploy_mode=1)
service.infer(prompt="...", seed=0, num_inference_steps=50,
              height=480, width=832, num_frames=124, output_path="output.mp4")
```

### 多图编辑

`edit_image` 统一使用 `list[PIL.Image]`：

```python
from PIL import Image
from services.qwen_image import QwenImageEdit2509

service = QwenImageEdit2509()
service.load_models(device="cuda", deploy_mode=1)
image = service.infer(
    prompt="生成这两个人的合影",
    negative_prompt="",
    seed=1,
    cfg_scale=4,
    num_inference_steps=40,
    height=1328,
    width=1024,
    edit_image=[Image.open("image_1.jpg"), Image.open("image_2.jpg")],
)
image.save("group.png")
```

### LoRA

大多数服务支持请求级 LoRA。传入 `(path, scale, keyword)` 列表即可，推理结束后服务会自动清空 LoRA，不影响下一个请求：

```python
image = service.infer(
    ...,
    loras=[("path/to/lora.safetensors", 1.0, "lora_keyword")],
)
```

## deploy_mode 说明

`deploy_mode` 是同一模型的不同部署策略，服务启动时通过 `load_models(deploy_mode=...)` 选择：

| deploy_mode | 权重 | 显存策略 | 适用场景 |
|-------------|------|----------|----------|
| 0 | 原始 BF16 / FP8 / NF4 权重 | 全部模型常驻 GPU | 显存足够时速度最快、精度最高 |
| 1 | 主循环 DiT 使用 EntroPack int8 量化，并 `torch.compile` | 量化权重常驻 GPU | 显存紧张、需要更高并发时 |
| 2 | 在 mode 1 基础上，把主循环之外的模型（文本编码器、VAE 等）offload 到 CPU | 主循环在 GPU，其余在 CPU | 显存最紧张时，速度换显存 |

不同服务的 mode 覆盖情况：

- 大多数图像、音频服务支持 mode 0 / 1 / 2；
- `MiniMaxH3FL2VA`、`MiniMaxH3Ref2VA` 支持 mode 0（NF4 权重）和 mode 1（int8 量化）；
- `MiniMaxH3Image` 只提供 mode 0（权重已经是 EntroPack 8bit / int8，加载后直接编译）。

`run_tests` 会从 mode 0 开始逐个尝试，直到某个 mode 抛出 `NotImplementedError` 为止。

## 在 H20 单卡集群上部署

### 单节点启动

H20 单卡机器的典型启动方式：

```shell
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/diffsynth_models/
export CUDA_VISIBLE_DEVICES=0
# 下面 FastAPI 示例中的 app.py
uvicorn app:app --host 0.0.0.0 --port 8000
```

`services/` 只提供模型服务类，不绑定具体的 HTTP 框架。AIGC 专区的网关会在服务启动时调用 `load_models`，收到请求后调用 `infer`。如果需要自己起一个最小 HTTP 服务，可以使用 FastAPI：

```python
import io
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from services.qwen_image import QwenImage

class Request(BaseModel):
    prompt: str
    negative_prompt: str = ""
    seed: int = 0
    num_inference_steps: int = 40
    height: int = 1024
    width: int = 1024

app = FastAPI()
service = QwenImage()

@app.on_event("startup")
def startup():
    service.load_models(device="cuda", deploy_mode=1)
    # 预热：编译、显存分配、CUDA kernel 初始化都在这里完成
    service.infer(prompt="warm up", seed=0, cfg_scale=4,
                  num_inference_steps=4, height=1024, width=1024)

@app.post("/generate")
def generate(request: Request):
    image = service.infer(**request.model_dump())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
```

启动：

```shell
pip install fastapi uvicorn
CUDA_VISIBLE_DEVICES=0 uvicorn app:app --host 0.0.0.0 --port 8000
```

### 集群部署建议

- **一个 GPU 一个进程。** H20 单卡机器只暴露一张卡；如果一台机器有多张卡，用 `CUDA_VISIBLE_DEVICES` 给每个进程分配一张卡，不要在一个进程里做多卡并行，模型服务本身不需要分布式。
- **模型缓存放共享存储。** 所有节点设置相同的 `DIFFSYNTH_MODEL_BASE_PATH`，首次启动时下载一次，之后各节点直接读缓存。
- **启动时预热。** `load_models` 之后先跑一次小分辨率推理，把 `torch.compile`、CuPy kernel 和显存池初始化好，避免第一个真实请求超时。
- **按显存选 deploy_mode。** 先用 `run_tests` 在目标机器上测出每个模型的显存占用，再决定 mode。H20 96GB 下，大部分图像模型在 mode 0 可以放下，视频模型和大文本编码器模型通常需要 mode 1 或 mode 2。
- **控制并发。** 单个 GPU 进程建议串行处理请求（并发数为 1），通过网关排队；同时跑多个请求会争抢显存，容易 OOM。
- **健康检查。** 健康检查只调用一个空的 `infer` 或检查 `service.pipe` 是否已加载，不要在健康检查里加载模型。
- **优雅退出。** 进程退出前把 `service.pipe = None`，释放显存和 CPU 内存。

### 模型与节点映射

AIGC 专区通常按模型维度做路由：网关记录“模型 → 服务实例地址”，每个实例在启动时注册自己支持的模型和服务类。推荐的部署方式是：

| 模型 | 单卡部署 | 说明 |
|------|----------|------|
| 图像模型（Qwen-Image、FLUX.2、Z-Image、Fibo、MOSS、Krea 等） | 1 卡 / 实例 | 优先 mode 0，显存不足时 mode 1 |
| 视频模型（MiniMax-H3） | 1 卡 / 实例 | 建议 mode 1 或 mode 2 |
| 音频模型（ACE-Step、MiniMax-Music3） | 1 卡 / 实例 | 可与图像模型混部在不同节点 |
| MiniMax-H3-Image | 1 卡 / 实例 | 只用 mode 0 |

节点故障时，网关把请求重新路由到其他同模型实例即可；模型服务本身无状态（LoRA 在请求结束后会清空）。

## 运行测试

`services/run_tests.py` 是统一入口：

```shell
# 列出所有服务
python -m services.run_tests --list

# 测试单个服务
CUDA_VISIBLE_DEVICES=0 python -m services.run_tests --class QwenImage --repeats 1

# 测试全部服务（每个服务一个子进程，互不影响）
CUDA_VISIBLE_DEVICES=0 python -m services.run_tests --all --repeats 1

# 只测试部分服务
CUDA_VISIBLE_DEVICES=0 python -m services.run_tests --only QwenImage QwenImageEdit Flux2Klein4B --repeats 1
```

参数说明：

- `--repeats`：每个样例重复推理的次数，默认 1。正式压测可以设为 3；
- `--modes`：只测试指定的 deploy_mode，例如 `--modes 0 1`；不传时逐个尝试直到 `NotImplementedError`；
- `--device`：`torch.device`，默认 `cuda`；
- `--timeout`：单个服务的超时时间，默认 6 小时；
- `ALLOW_TORCH_COMPILE=1`：强制启用 `torch.compile`（需要 Triton）。

输出目录：

```
data/diffsynth_service_tests/
├── assets/                  # 测试输入文件
├── logs/                    # 每个服务的完整日志和 summary.json
├── videos/                  # 视频服务的输出（由 output_path 指定）
└── <Service>_mode<M>_sample<S>_run<R>.png|.pt   # run_tests 生成的图片/音频输出
```

`run_tests` 会为每个 `deploy_mode` 输出两张表：

- **Table 1**：每个阶段（Waiting / Loading / Inference / Maximum）的 GPU 显存（VRAM GB）和进程内存（Memory GB）；
- **Table 2**：每个样例、每次运行的推理耗时。

其中 `Inference` 是推理阶段的峰值，`Maximum` 是整个流程（含加载）的峰值。部署时以 `Maximum` 为准选择显存规格。

### 测试输入文件

部分样例需要输入图片或音视频。所有输入统一放在 `data/diffsynth_service_tests/assets/`：

```
assets/
├── fibo/                 # 编辑输入图和蒙版
├── flux2/                # 图片编辑输入图
├── hidream_o1/           # 图片编辑输入图
├── minimax_h3/           # 首尾帧、参考图、参考视频、参考音频
├── minimax_h3_image/     # 图片编辑输入图
├── qwen_image/           # 单图/多图编辑输入图
└── qwen_image_21/        # 双人合影编辑输入图（RGBA）
```

这些文件来自 DiffSynth-Studio 的示例数据集，或由示例脚本中的“先生成再编辑”流程预先生成。重新生成：

```shell
CUDA_VISIBLE_DEVICES=0 python data/diffsynth_service_tests/tools/make_assets.py
```

`run_tests.py` 和 `make_assets.py` 都应在 **DiffSynth-Studio 仓库根目录**下执行，因为服务代码中的输入路径是相对仓库根目录的。

## 常见问题

**Q：首次请求特别慢？**
A：`load_models` 只加载权重；`torch.compile`、CuPy kernel 编译和显存池初始化发生在第一次推理。服务启动后先跑一次预热推理。

**Q：显存不够（OOM）？**
A：优先降低 `deploy_mode` 的显存占用：mode 0 → mode 1 → mode 2。仍然不够时降低分辨率、帧数或 `num_inference_steps`。`run_tests` 输出的 `Maximum` 列可以直接用来判断。

**Q：`torch.compile` 报 `TritonMissing`？**
A：当前环境没有 Triton。设置 `TORCHDYNAMO_DISABLE=1` 以 eager 模式运行，量化仍然生效；或者安装 Triton。

**Q：EntroPack 报 `cupy is not installed`？**
A：安装 CuPy：`pip install cupy`。

**Q：NF4 模型报 `bitsandbytes is required`？**
A：安装 `bitsandbytes`：`pip install bitsandbytes`。

**Q：MiniMax-Music3 的 mode 2 特别慢？**
A：mode 2 会把自回归语言模型 offload 到 CPU，而音乐生成需要逐 token 解码，CPU 推理会慢到不可接受。MiniMax-Music3 建议使用 mode 0 或 mode 1。

**Q：模型下载很慢或没有网络？**
A：提前把模型放到 `DIFFSYNTH_MODEL_BASE_PATH` 下，或设置 `DIFFSYNTH_SKIP_DOWNLOAD=true` 只使用本地缓存。

**Q：视频服务为什么不返回文件？**
A：`MiniMaxH3FL2VA`、`MiniMaxH3Ref2VA` 的 `infer` 直接把视频写到 `output_path`，返回值是 `None`。请求方传入自己的输出路径即可。

## 相关文档

- [模型推理](./Model_Inference.md)
- [显存管理](./VRAM_management.md)
- [模型量化](./Quantization.md)
- [环境变量](./Environment_Variables.md)
- [GPU/NPU 支持](./GPU_support.md)
