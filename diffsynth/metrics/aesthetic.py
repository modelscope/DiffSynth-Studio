import torch
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.aesthetic import AestheticModel
from .base import Metric
from transformers import CLIPImageProcessor

class AestheticMetric(Metric):
    def __init__(self, model: AestheticModel):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="Aesthetic/model.safetensors"),
        processor_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="Aesthetic/"),
        torch_dtype: torch.dtype = None,
        device: torch.device = get_device_type(),
        processor_kwargs: dict = None,
        vram_limit: float = None,
    ):

        processor_kwargs = processor_kwargs or {}
        model_pool = cls.download_and_load_models([model_config], torch_dtype=torch_dtype, device=device, vram_limit=vram_limit)
        model = model_pool.fetch_model("image_metrics_aesthetic")
        processor_config.download_if_necessary()
        model.processor = CLIPImageProcessor.from_pretrained(processor_config.path, **processor_kwargs)
        model.layers = model.layers.float()
        model = model.eval()
        return cls(model)

    @torch.no_grad()
    def score(self, images):
        scores = self.model(images)
        return self.tensor_to_list(scores)

    def compute(self, images):
        return self.score(images)

    def forward(self, images):
        return self.score(images)
