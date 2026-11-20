# Deploying Inference Services

The `services/` directory contains the model inference services used by the ModelScope AIGC Zone. It wraps DiffSynth-Studio pipelines in a uniform *model service* interface and targets **clusters built from single-GPU H20 machines**: each machine has one GPU, and the cluster scales horizontally by adding machines. Every machine runs one or more service instances.

Unlike the one-shot scripts under `examples/`, each class in `services/` is a **long-running service**:

- `load_models(device, deploy_mode)` loads the model when the service starts and keeps it resident on the GPU;
- `infer(**kwargs)` handles one request and returns an image, an audio tensor, or writes a video to `output_path`;
- `example_inputs()` returns test samples aligned with the corresponding inference script under `examples/`;
- `run_tests()` benchmarks every `deploy_mode` (load + inference) and prints VRAM, memory, and latency tables.

## Services

| Module | Service class | Model | Task | Output | deploy_mode |
|--------|---------------|-------|------|--------|-------------|
| `ace_step` | `AceStep15XLSFT` | ACE-Step 1.5 XL SFT | text-to-music | audio tensor | 0 / 1 / 2 |
| `fibo` | `Fibo20Dev` | Fibo 2.0 dev | text-to-image, image edit, masked edit | `PIL.Image` | 0 / 1 / 2 |
| `fibo` | `Fibo20EditTurbo` | Fibo 2.0 edit turbo | image edit, masked edit | `PIL.Image` | 0 / 1 / 2 |
| `flux2` | `Flux2Klein4B` | FLUX.2-klein-4B | text-to-image, image edit | `PIL.Image` | 0 / 1 / 2 |
| `flux2` | `Flux2KleinBase4B` | FLUX.2-klein-base-4B | text-to-image, image edit | `PIL.Image` | 0 / 1 / 2 |
| `flux2` | `Flux2Klein9B` | FLUX.2-klein-9B | text-to-image, image edit | `PIL.Image` | 0 / 1 / 2 |
| `flux2` | `Flux2KleinBase9B` | FLUX.2-klein-base-9B | text-to-image, image edit | `PIL.Image` | 0 / 1 / 2 |
| `hidream_o1` | `HiDreamO1Image` | HiDream-O1-Image | text-to-image, multi-reference edit | `PIL.Image` | 0 / 1 / 2 |
| `krea2_turbo` | `Krea2Turbo` | Krea-2-Turbo | text-to-image | `PIL.Image` | 0 / 1 / 2 |
| `minimax_h3` | `MiniMaxH3FL2VA` | MiniMax-H3 FL2VA | text/first-last-frame-to-audio-video | writes mp4 | 0 / 1 |
| `minimax_h3` | `MiniMaxH3Ref2VA` | MiniMax-H3 Ref2VA | reference image/video/audio-to-audio-video | writes mp4 | 0 / 1 |
| `minimax_h3_image` | `MiniMaxH3Image` | MiniMax-H3-Image | text-to-image, image edit | `PIL.Image` | 0 |
| `minimax_music` | `MiniMaxMusic3` | MiniMax-Music3 | text-to-music | audio tensor | 0 / 1 / 2 |
| `moss_image` | `MOSSImageSFT` | MOSS-Image SFT | text-to-image | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `QwenImage` | Qwen-Image | text-to-image (optional Context Control) | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `QwenImage2512` | Qwen-Image-2512 | text-to-image | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `QwenImageEdit` | Qwen-Image-Edit | single-image edit | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `QwenImageEdit2509` | Qwen-Image-Edit-2509 | multi-image edit | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `QwenImageEdit2511` | Qwen-Image-Edit-2511 | multi-image edit | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image` | `FireRedImageEdit` | FireRed-Image-Edit-1.0 | multi-image edit | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image_21` | `QwenImage21` | Qwen-Image-2.1 | text-to-image (RGBA), multi-image edit | `PIL.Image` | 0 / 1 / 2 |
| `qwen_image_21` | `QwenImage21Turbo` | Qwen-Image-2.1-Turbo | text-to-image (RGBA), multi-image edit | `PIL.Image` | 0 / 1 / 2 |
| `z_image` | `ZImageTurbo` | Z-Image-Turbo | text-to-image | `PIL.Image` | 0 / 1 / 2 |
| `z_image` | `ZImage` | Z-Image | text-to-image | `PIL.Image` | 0 / 1 / 2 |

## Environment

Install:

```shell
pip install -U diffsynth cupy bitsandbytes
```

DiffSynth-Studio downloads models from ModelScope automatically. Put the weights on shared storage and point the environment variable at the cache directory:

```shell
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/models/
```

On H20 GPUs, we recommend installing Sage Attention and registering it with the framework. The following code must run before `diffsynth` is imported for the first time:

```python
from services.utils import register_attention
register_attention()
```

## Quick Start

Text-to-image with Qwen-Image:

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

Call `load_models` once at startup. Every request then calls `infer` directly; the weights stay resident on the GPU.

<details>
<summary>Show more</summary>

Handling the different output types:

```python
# Image: infer returns a PIL.Image
image.save("output.png")

# Audio: infer returns a torch.Tensor; read the sample rate from the pipeline vocoder
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

# Video: infer returns nothing and writes an mp4 (with audio) to output_path
from services.minimax_h3 import MiniMaxH3FL2VA
service = MiniMaxH3FL2VA()
service.load_models(device="cuda", deploy_mode=1)
service.infer(prompt="...", seed=0, num_inference_steps=50,
              height=480, width=832, num_frames=124, output_path="output.mp4")
```

`edit_image` is always a `list[PIL.Image]`:

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

Most services accept per-request LoRAs as a list of `(path, scale, keyword)`. The service clears them after inference, so they never leak into the next request:

```python
image = service.infer(
    ...,
    loras=[("path/to/lora.safetensors", 1.0, "lora_keyword")],
)
```

</details>

## deploy_mode

`deploy_mode` selects a deployment strategy for the same model and is passed to `load_models(deploy_mode=...)` at startup:

| deploy_mode | Weights | Memory strategy | When to use |
|-------------|---------|-----------------|-------------|
| 0 | original BF16 / FP8 / NF4 | all models resident on GPU | enough VRAM; fastest and most accurate |
| 1 | main-loop DiT quantized with EntroPack int8, plus `torch.compile` | quantized weights resident on GPU | tight VRAM or higher concurrency |
| 2 | mode 1 plus CPU offload of models outside the main loop (text encoder, VAE, ...) | main loop on GPU, the rest on CPU | most VRAM-constrained; trades speed for memory |

Coverage differs per service:

- most image and audio services support modes 0 / 1 / 2;
- `MiniMaxH3FL2VA` and `MiniMaxH3Ref2VA` are too large and only support mode 0 (NF4 weights) and mode 1 (int8 quantization);
- `MiniMaxH3Image` provides only mode 0 for more efficient inference (weights are already EntroPack 8-bit / int8 and the pipeline is compiled after loading).

`run_tests` tries modes starting from 0 and reports the compute resource usage of every `deploy_mode`.

## Deploying on a Single-GPU H20 Cluster

### Starting one node

A typical startup command on a single-GPU H20 machine:

```shell
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/models/
export CUDA_VISIBLE_DEVICES=0
# app.py from the FastAPI example below
uvicorn app:app --host 0.0.0.0 --port 8000
```

`services/` provides model service classes and deliberately does not bind an HTTP framework. The AIGC Zone gateway calls `load_models` at startup and `infer` for each request. For a minimal HTTP service of your own, use FastAPI:

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
    # Warm-up: compilation, VRAM allocation and CUDA kernel init happen here
    service.infer(prompt="warm up", seed=0, cfg_scale=4,
                  num_inference_steps=4, height=1024, width=1024)

@app.post("/generate")
def generate(request: Request):
    image = service.infer(**request.model_dump())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
```

Launch it with:

```shell
pip install fastapi uvicorn
CUDA_VISIBLE_DEVICES=0 uvicorn app:app --host 0.0.0.0 --port 8000
```

### Cluster recommendations

- **One process per GPU.** A single-GPU H20 machine exposes one device. On multi-GPU machines, assign one device per process with `CUDA_VISIBLE_DEVICES`.
- **Share the model cache.** Set the same `DIFFSYNTH_MODEL_BASE_PATH` on every node. The first start downloads once; later starts read the cache directly.
- **Warm up at startup.** After `load_models`, run one small inference to initialize `torch.compile`, CuPy kernels and the VRAM pool; otherwise the first real request may time out.
- **Pick deploy_mode by VRAM.** Run `run_tests` on the target hardware first, then choose the mode. On a 96GB H20 most image models fit in mode 0, while video models and large text-encoder models usually need mode 1 or 2.
- **Limit concurrency.** Process requests serially per GPU process (concurrency 1) and queue at the gateway; concurrent requests compete for VRAM and easily OOM.
- **Health checks.** Check that `service.pipe` is loaded or run an empty `infer`; never load weights inside a health check.
- **Graceful shutdown.** Set `service.pipe = None` before exit to release GPU and CPU memory.

## Running Tests

`services/run_tests.py` is the unified entry point:

```shell
# List every service
python -m services.run_tests --list

# Test a single service
CUDA_VISIBLE_DEVICES=0 python -m services.run_tests --class QwenImage --repeats 1

# Test every service (one subprocess per service, isolated failures)
CUDA_VISIBLE_DEVICES=0 python -m services.run_tests --all --repeats 1

# Test a subset
CUDA_VISIBLE_DEVICES=0 python -m services.run_tests --only QwenImage QwenImageEdit Flux2Klein4B --repeats 1
```

Options:

- `--repeats`: inference repeats per sample, default 1 (use 3 for a formal benchmark);
- `--modes`: test only the given deploy modes, e.g. `--modes 0 1`; without it every mode is tried until `NotImplementedError`;
- `--device`: `torch.device`, default `cuda`;
- `--timeout`: per-service timeout, default 6 hours;
- `ALLOW_TORCH_COMPILE=1`: force `torch.compile` (requires Triton).

Output layout:

```
data/diffsynth_service_tests/
├── assets/                  # test inputs
├── logs/                    # per-service logs and summary.json
├── videos/                  # outputs of video services (writer path from output_path)
└── <Service>_mode<M>_sample<S>_run<R>.png|.pt   # image/audio outputs from run_tests
```

`run_tests` prints two tables per `deploy_mode`:

- **Table 1**: GPU VRAM and process memory (GB) for every phase — Waiting / Loading / Inference / Maximum;
- **Table 2**: inference latency per sample and run.

`Inference` is the peak during inference; `Maximum` covers the whole flow including loading. Use `Maximum` when sizing a GPU.

### Test inputs

Download the test input files:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_service_tests --local_dir ./data/diffsynth_service_tests
```

## FAQ

**Q: The first request is very slow.**
A: `load_models` only loads weights; `torch.compile`, CuPy kernel compilation and VRAM pool initialization happen on the first inference. Run a warm-up inference after startup.

**Q: Out of memory?**
A: Move down the deploy modes: 0 → 1 → 2. If that is not enough, reduce resolution, frame count, or `num_inference_steps`. The `Maximum` column from `run_tests` tells you the required VRAM.

**Q: `torch.compile` raises `TritonMissing`.**
A: Triton is unavailable. Set `TORCHDYNAMO_DISABLE=1` to run in eager mode (quantization still applies), or install Triton.

**Q: EntroPack reports `cupy is not installed`.**
A: Install CuPy: `pip install cupy`.

**Q: An NF4 model reports `bitsandbytes is required`.**
A: Install `bitsandbytes`: `pip install bitsandbytes`.

**Q: MiniMax-Music3 mode 2 is extremely slow.**
A: Mode 2 offloads its autoregressive language model to the CPU, and music generation decodes token by token, so CPU inference is impractically slow. Use mode 0 or mode 1 for MiniMax-Music3.

**Q: Model download is slow or there is no network.**
A: Pre-populate `DIFFSYNTH_MODEL_BASE_PATH`, then set `DIFFSYNTH_SKIP_DOWNLOAD=true` to use only the local cache and skip all download requests.

## See Also

- [Model Inference](./Model_Inference.md)
- [VRAM Management](./VRAM_management.md)
- [Model Quantization](./Quantization.md)
- [Environment Variables](./Environment_Variables.md)
- [GPU/NPU Support](./GPU_support.md)
