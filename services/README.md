# Deploying Inference Services

`services/` contains the inference service code used by the ModelScope AIGC Zone. Each class wraps a DiffSynth-Studio pipeline as a long-running model service for **single-GPU H20 machines**, exposing a stable `load_models` / `infer` / `run_tests` interface and multiple `deploy_mode` memory strategies.

完整的使用说明和部署建议请参考文档：

- [部署推理服务](../docs/zh/Pipeline_Usage/Inference_Service.md)
- [Deploying Inference Services](../docs/en/Pipeline_Usage/Inference_Service.md)

## Quick start

```shell
conda create -n diffsynth_service --clone <your-pytorch-env> -y
conda activate diffsynth_service
pip install -e .

export DIFFSYNTH_MODEL_BASE_PATH=/path/to/diffsynth_models/
CUDA_VISIBLE_DEVICES=0 python -m services.run_tests --list
CUDA_VISIBLE_DEVICES=0 python -m services.run_tests --class QwenImage --repeats 1
```

```python
from services.qwen_image import QwenImage

service = QwenImage()
service.load_models(device="cuda", deploy_mode=0)
image = service.infer(
    prompt="精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。",
    seed=0,
    num_inference_steps=40,
    height=1472,
    width=1104,
)
image.save("output.png")
```

Test inputs live in [`data/diffsynth_service_tests/assets`](../data/diffsynth_service_tests/assets), and benchmark logs/outputs are written to [`data/diffsynth_service_tests`](../data/diffsynth_service_tests).
