from pathlib import Path
from typing import Union
import torch
from ..core import ModelConfig

class Metric(torch.nn.Module):
    @staticmethod
    def tensor_to_list(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()
        if not isinstance(value, list):
            return [value]
        return value

    @staticmethod
    def tensor_to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

    @staticmethod
    def resolve_model_config(config: Union[ModelConfig, str, Path]):
        if isinstance(config, (str, Path)):
            config = ModelConfig(path=str(config))
        if config is None:
            return None
        config.download_if_necessary()
        return config

    @staticmethod
    def local_or_modelscope_config(model_id: str, origin_file_pattern: str = ""):
        local_path = Path("./models") / model_id
        if local_path.exists():
            return ModelConfig(path=str(local_path))
        return ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern)
