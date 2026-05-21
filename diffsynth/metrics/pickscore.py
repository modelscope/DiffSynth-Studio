from transformers import AutoProcessor
import torch
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.pickscore import PickScoreModel
from .base import Metric

class PickScoreMetric(Metric):
    def __init__(self, model: PickScoreModel):
        super().__init__()
        self.model = model
        
    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="PickScore/model.safetensors"),
        processor_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="PickScore/"),
        torch_dtype: torch.dtype = None,
        device: torch.device = get_device_type(),
        max_length: int = 77,
        processor_kwargs: dict = None,
        vram_limit: float = None,
    ):
    
        processor_kwargs = processor_kwargs or {}
        model_pool = cls.download_and_load_models([model_config], torch_dtype=torch_dtype, device=device, vram_limit=vram_limit)
        model = model_pool.fetch_model("image_metrics_clip_hf")
        processor_config.download_if_necessary()
        processor = AutoProcessor.from_pretrained(processor_config.path, **processor_kwargs)
        model = PickScoreModel(model=model, processor=processor, max_length=max_length).eval()
        return cls(model)

    @torch.no_grad()
    def score(
        self,
        prompt: str | list[str],
        images,
    ):
        scores = self.model(prompt, images)
        return self.tensor_to_list(scores)

    @torch.no_grad()
    def probabilities(
        self,
        prompt: str | list[str],
        images,
    ):
        scores = self.model(prompt, images)
        probabilities = torch.softmax(scores, dim=-1)
        return self.tensor_to_list(probabilities)

    def calc_probs(self, prompt: str | list[str], images):
        return self.probabilities(prompt, images)

    def compute(self, prompt: str | list[str], images):
        return self.score(prompt, images)

    def forward(self, prompt: str | list[str], images):
        return self.score(prompt, images)
