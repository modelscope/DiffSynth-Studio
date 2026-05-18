from typing import Union
import torch
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.hpsv3 import HPSv3Model
from .base import Metric


class HPSv3Metric(Metric):
    def __init__(self, model: HPSv3Model):
        super().__init__()
        self.model = model

    @staticmethod
    def default_model_config():
        return HPSv3Metric.local_or_modelscope_config("MizzenAI/HPSv3")

    @staticmethod
    def default_base_model_config():
        return HPSv3Metric.local_or_modelscope_config("Qwen/Qwen2-VL-7B-Instruct")

    @classmethod
    def from_pretrained(
        cls,
        model_config: Union[ModelConfig, str] = None,
        base_model_config: Union[ModelConfig, str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        output_dim: int = 2,
        score_index: int = 0,
        use_special_tokens: bool = True,
        max_pixels: int = 256 * 28 * 28,
        min_pixels: int = 256 * 28 * 28,
        model_kwargs: dict = None,
        processor_kwargs: dict = None,
    ):
        model_config = cls.default_model_config() if model_config is None else model_config
        base_model_config = cls.default_base_model_config() if base_model_config is None else base_model_config
        model_config = cls.resolve_model_config(model_config)
        base_model_config = cls.resolve_model_config(base_model_config)
        model = HPSv3Model.from_pretrained(
            model_path=model_config.path,
            base_model_path=base_model_config.path,
            torch_dtype=torch_dtype,
            device=device,
            output_dim=output_dim,
            score_index=score_index,
            use_special_tokens=use_special_tokens,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
        )
        return cls(model)

    @torch.no_grad()
    def score(self, prompt: Union[str, list[str]], images):
        scores = self.model(prompt, images)
        return self.tensor_to_list(scores)

    def calc_scores(self, prompt: Union[str, list[str]], images):
        return self.score(prompt, images)

    def forward(self, prompt: Union[str, list[str]], images):
        return self.score(prompt, images)
