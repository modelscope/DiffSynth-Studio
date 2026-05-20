import torch

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.fid import FIDModel
from .base import Metric


class FIDMetric(Metric):
    def __init__(self, model: FIDModel):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="FID/model.safetensors"),
        device: torch.device = get_device_type(),
        batch_size: int = 16,
        num_workers: int = 0,
        vram_limit: float = None,
    ):
        model_pool = cls.download_and_load_models([model_config], torch_dtype=torch.float32, device=device, vram_limit=vram_limit)
        model = model_pool.fetch_model("image_metrics_fid_inception")
        model = FIDModel(model=model, device=device, batch_size=batch_size, num_workers=num_workers)
        return cls(model)

    @torch.no_grad()
    def compute(self, reference_images, generated_images, batch_size: int = None, num_workers: int = None):
        score = self.model.compute(reference_images, generated_images, batch_size=batch_size, num_workers=num_workers)
        return score.detach().cpu().item() if torch.is_tensor(score) else float(score)

    def statistics(self, images, batch_size: int = None, num_workers: int = None):
        return self.model.statistics(images, batch_size=batch_size, num_workers=num_workers)

    def forward(self, reference_images, generated_images):
        return self.compute(reference_images, generated_images)
