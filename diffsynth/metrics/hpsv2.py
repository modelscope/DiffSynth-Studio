import torch
from transformers import AutoProcessor
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.hpsv2 import HPSv2Model
from .base import Metric

class HPSv2Metric(Metric):
    def __init__(self, model: HPSv2Model):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="HPSv2/model.safetensors"),
        processor_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="HPSv2/"),
        torch_dtype: torch.dtype = None,
        device: torch.device = get_device_type(),
        processor_kwargs: dict = None,
        vram_limit: float = None,
    ):

        processor_kwargs = processor_kwargs or {}
        model_pool = cls.download_and_load_models([model_config], torch_dtype=torch_dtype, device=device, vram_limit=vram_limit)
        model = model_pool.fetch_model("image_metrics_hpsv2")
        processor_config.download_if_necessary()
        processor = AutoProcessor.from_pretrained(processor_config.path, **processor_kwargs)
        model = HPSv2Model(model=model, processor=processor).eval()
        return cls(model)

    @torch.no_grad()
    def score(self, prompt: str | list[str], images):
        scores = self.model(prompt, images)
        return self.tensor_to_list(scores)

    def compute(self, prompt: str | list[str], images):
        return self.score(prompt, images)

    def forward(self, prompt: str | list[str], images):
        return self.score(prompt, images)
