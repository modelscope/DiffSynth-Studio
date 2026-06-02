import torch

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.lpips import LPIPSModel, LPIPS_NET_CHOICES, LPIPSCompute
from .base import Metric


_LPIPS_DEFAULT_FILES = {
    "alex": "LPIPS/alexnet.safetensors",
    "vgg": "LPIPS/vgg.safetensors",
    "squeeze": "LPIPS/squeezenet.safetensors",
}

_LPIPS_MODEL_NAMES = {
    "alex": "image_metrics_lpips_alex",
    "vgg": "image_metrics_lpips_vgg",
    "squeeze": "image_metrics_lpips_squeeze",
}


class LPIPSMetric(Metric):
    def __init__(self, model: LPIPSCompute):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        net: str = "alex",
        model_config: ModelConfig = None,
        device: torch.device = get_device_type(),
        batch_size: int = 16,
        target_size: int = 512,
        vram_limit: float = None,
    ):
        if net not in LPIPS_NET_CHOICES:
            raise ValueError(f"net must be one of {LPIPS_NET_CHOICES}, got {net!r}")
        if model_config is None:
            model_config = ModelConfig(
                model_id="DiffSynth-Studio/ImageMetrics",
                origin_file_pattern=_LPIPS_DEFAULT_FILES[net],
            )
        model_pool = cls.download_and_load_models([model_config], torch_dtype=torch.float32, device=device, vram_limit=vram_limit)
        backbone = model_pool.fetch_model(_LPIPS_MODEL_NAMES[net])
        if backbone is None:
            raise RuntimeError(
                f"Failed to load LPIPS model for net={net!r}. The provided weights do not match the registered hash for {_LPIPS_MODEL_NAMES[net]}."
            )
        compute_model = LPIPSCompute(
            model=backbone,
            device=device,
            batch_size=batch_size,
            target_size=target_size,
        )
        return cls(compute_model)

    @torch.no_grad()
    def compute(self, image_a, image_b) -> float:
        return self.model.compute(image_a, image_b)

    def forward(self, image_a, image_b):
        return self.compute(image_a, image_b)
