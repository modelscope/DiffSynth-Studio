from typing import Union
import torch
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
        model_config: Union[ModelConfig, str] = "AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K",
        processor_config: Union[ModelConfig, str] = None,
        torch_dtype: torch.dtype = None,
        device: Union[str, torch.device] = get_device_type(),
        max_length: int = 77,
        model_kwargs: dict = None,
        processor_kwargs: dict = None,
    ):
        model_config = cls.resolve_model_config(model_config)
        processor_config = cls.resolve_model_config(processor_config) if processor_config is not None else model_config
        model = CLIPModel.from_pretrained(
            model_path=model_config.path,
            processor_path=processor_config.path,
            torch_dtype=torch_dtype,
            device=device,
            max_length=max_length,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
        )
        return cls(model)

    @torch.no_grad()
    def score(
        self,
        prompt: Union[str, list[str]],
        images,
    ):
        scores = self.model(prompt, images)
        return self.tensor_to_list(scores)

    @torch.no_grad()
    def similarity_matrix(
        self,
        prompt: Union[str, list[str]],
        images,
    ):
        scores = self.model.similarity_matrix(prompt, images)
        return self.tensor_to_list(scores)

    def compute(self, prompt: Union[str, list[str]], images):
        return self.score(prompt, images)

    def forward(self, prompt: Union[str, list[str]], images):
        return self.score(prompt, images)
