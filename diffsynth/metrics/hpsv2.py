from typing import Union
import torch
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
        model_config: Union[ModelConfig, str] = "AI-ModelScope/HPSv2",
        processor_config: Union[ModelConfig, str] = "AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K",
        version: str = "v2.0",
        torch_dtype: torch.dtype = None,
        device: Union[str, torch.device] = get_device_type(),
        model_kwargs: dict = None,
        processor_kwargs: dict = None,
    ):
        model_config = cls.resolve_model_config(model_config)
        processor_config = cls.resolve_model_config(processor_config)
        model = HPSv2Model.from_pretrained(
            model_path=model_config.path,
            processor_path=processor_config.path,
            version=version,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
        )
        return cls(model)

    @torch.no_grad()
    def score(self, prompt: Union[str, list[str]], images):
        scores = self.model(prompt, images)
        return self.tensor_to_list(scores)

    def compute(self, prompt: Union[str, list[str]], images):
        return self.score(prompt, images)

    def forward(self, prompt: Union[str, list[str]], images):
        return self.score(prompt, images)
