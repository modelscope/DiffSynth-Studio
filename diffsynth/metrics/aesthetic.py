from typing import Union
import torch
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.aesthetic import AestheticModel
from .base import Metric

class AestheticMetric(Metric):
    def __init__(self, model: AestheticModel):
        super().__init__()
        self.model = model

    @staticmethod
    def default_model_config():
        return AestheticMetric.local_or_modelscope_config("AI-ModelScope/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE")

    @classmethod
    def from_pretrained(
        cls,
        model_config: Union[ModelConfig, str] = None,
        clip_config: Union[ModelConfig, str] = None,
        clip_processor_config: Union[ModelConfig, str] = None,
        torch_dtype: torch.dtype = None,
        device: Union[str, torch.device] = get_device_type(),
        clip_kwargs: dict = None,
        processor_kwargs: dict = None,
    ):
        model_config = cls.default_model_config() if model_config is None else model_config
        model_config = cls.resolve_model_config(model_config)
        clip_config = cls.resolve_model_config(clip_config) if clip_config is not None else None
        clip_processor_config = cls.resolve_model_config(clip_processor_config) if clip_processor_config is not None else clip_config
        model = AestheticModel.from_pretrained(
            model_path=model_config.path,
            clip_model_path=None if clip_config is None else clip_config.path,
            clip_processor_path=None if clip_processor_config is None else clip_processor_config.path,
            torch_dtype=torch_dtype,
            device=device,
            clip_kwargs=clip_kwargs,
            processor_kwargs=processor_kwargs,
        )
        return cls(model)

    @torch.no_grad()
    def score(self, images):
        scores = self.model(images)
        return self.tensor_to_list(scores)

    def calc_scores(self, images):
        return self.score(images)

    def forward(self, images):
        return self.score(images)
