import torch
from transformers import AutoProcessor
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.qwen_image_bench import QwenImageBenchModel
from .base import Metric
from transformers.utils import logging
logging.set_verbosity_error()


class QwenImageBenchMetric(Metric):
    def __init__(self, model: QwenImageBenchModel):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig = ModelConfig(
            model_id="Qwen/Qwen-Image-Bench",
            origin_file_pattern="model-*.safetensors",
        ),
        processor_config: ModelConfig = ModelConfig(
            model_id="Qwen/Qwen-Image-Bench",
            origin_file_pattern="",
        ),
        torch_dtype: torch.dtype = None,
        device: torch.device = get_device_type(),
        max_new_tokens: int = 4096,
        resize_long_edge: int = 1024,
        processor_kwargs: dict = None,
        vram_limit: float = None,
    ):
        processor_kwargs = processor_kwargs or {}
        model_pool = cls.download_and_load_models(
            [model_config],
            torch_dtype=torch_dtype or torch.bfloat16,
            device=device,
            vram_limit=vram_limit,
        )
        model = model_pool.fetch_model("image_metrics_qwen_image_bench")
        if model is None:
            raise ValueError("Cannot find model: image_metrics_qwen_image_bench")
        if hasattr(model, "model"):
            model = model.model

        processor_config.download_if_necessary()
        processor = AutoProcessor.from_pretrained(processor_config.path, **processor_kwargs)
        model = QwenImageBenchModel(
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
            resize_long_edge=resize_long_edge,
        ).eval()
        return cls(model)

    @torch.no_grad()
    def evaluate(self, prompt: str | list[str] | None, images, dimensions=None):
        return self.model(prompt, images, dimensions=dimensions)

    @torch.no_grad()
    def score(self, prompt: str | list[str] | None, images, dimensions=None):
        outputs = self.evaluate(prompt, images, dimensions=dimensions)
        return [self.model._primary_score(output) for output in outputs]

    def compute(self, prompt: str | list[str] | None, images, dimensions=None):
        return self.score(prompt, images, dimensions=dimensions)

    def forward(self, prompt: str | list[str] | None, images, dimensions=None):
        return self.score(prompt, images, dimensions=dimensions)
