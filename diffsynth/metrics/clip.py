import torch
from transformers import AutoProcessor
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.clip import CLIPModel
from .base import Metric

class CLIPMetric(Metric):
    def __init__(self, model: CLIPModel):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors"),
        processor_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="CLIP-ViT-H-14-laion2B-s32B-b79K/"),
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
        model = CLIPModel(model=model, processor=processor, max_length=max_length).eval()
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
    def similarity_matrix(
        self,
        prompt: str | list[str],
        images,
    ):
        scores = self.model.similarity_matrix(prompt, images)
        return self.tensor_to_list(scores)

    def compute(self, prompt: str | list[str], images):
        return self.score(prompt, images)

    def forward(self, prompt: str | list[str], images):
        return self.score(prompt, images)
