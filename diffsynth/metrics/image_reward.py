import os
from pathlib import Path
from typing import Union

import torch
from modelscope import snapshot_download

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.image_reward import ImageRewardModel
from .base import Metric

class ImageRewardMetric(Metric):
    BERT_TOKENIZER_MODEL_ID = "AI-ModelScope/bert-base-uncased"
    BERT_TOKENIZER_FILES = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    ]

    def __init__(self, model: ImageRewardModel):
        super().__init__()
        self.model = model

    @staticmethod
    def default_model_config():
        return ImageRewardMetric.local_or_modelscope_config("ZhipuAI/ImageReward")

    @staticmethod
    def default_tokenizer_config():
        local_path = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models")) / ImageRewardMetric.BERT_TOKENIZER_MODEL_ID
        if all((local_path / filename).exists() for filename in ImageRewardMetric.BERT_TOKENIZER_FILES):
            return ModelConfig(path=str(local_path))
        return ModelConfig(path=str(ImageRewardMetric.download_default_tokenizer()))

    @staticmethod
    def download_default_tokenizer():
        local_path = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models")) / ImageRewardMetric.BERT_TOKENIZER_MODEL_ID
        if all((local_path / filename).exists() for filename in ImageRewardMetric.BERT_TOKENIZER_FILES):
            return local_path
        local_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            ImageRewardMetric.BERT_TOKENIZER_MODEL_ID,
            local_dir=str(local_path),
            allow_file_pattern=ImageRewardMetric.BERT_TOKENIZER_FILES,
            local_files_only=False,
        )
        missing = [filename for filename in ImageRewardMetric.BERT_TOKENIZER_FILES if not (local_path / filename).exists()]
        if missing:
            raise FileNotFoundError(f"Missing ImageReward tokenizer files under {local_path}: {missing}")
        return local_path

    @staticmethod
    def _as_directory_path(path):
        if isinstance(path, list):
            if len(path) == 0:
                raise FileNotFoundError("Downloaded tokenizer files are empty.")
            return str(Path(path[0]).parent)
        return path

    @classmethod
    def from_pretrained(
        cls,
        model_config: Union[ModelConfig, str] = None,
        med_config: Union[ModelConfig, str] = None,
        tokenizer_config: Union[ModelConfig, str] = None,
        torch_dtype: torch.dtype = None,
        device: Union[str, torch.device] = get_device_type(),
        max_length: int = 35,
        model_kwargs: dict = None,
        tokenizer_kwargs: dict = None,
    ):
        model_config = cls.default_model_config() if model_config is None else model_config
        tokenizer_config = cls.default_tokenizer_config() if tokenizer_config is None else tokenizer_config
        model_config = cls.resolve_model_config(model_config)
        med_config = cls.resolve_model_config(med_config) if med_config is not None else None
        tokenizer_config = cls.resolve_model_config(tokenizer_config)
        model = ImageRewardModel.from_pretrained(
            model_path=model_config.path,
            med_config_path=None if med_config is None else med_config.path,
            tokenizer_path=cls._as_directory_path(tokenizer_config.path),
            torch_dtype=torch_dtype,
            device=device,
            max_length=max_length,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
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
