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
    def resolve_model_config(config: Union[ModelConfig, str, Path], origin_file_pattern: str = ""):
        if config is None:
            return None
        if isinstance(config, Path):
            config = ModelConfig(path=str(config))
        elif isinstance(config, str):
            path = Path(config).expanduser()
            if path.exists() or path.is_absolute() or config.startswith(("./", "../", "~")) or ("/" not in config and path.suffix):
                config = ModelConfig(path=str(path))
            else:
                local_path = Path("./models") / config
                if not origin_file_pattern and local_path.exists():
                    config = ModelConfig(path=str(local_path))
                else:
                    config = ModelConfig(model_id=config, origin_file_pattern=origin_file_pattern)
        if config is None:
            return None
        config.download_if_necessary()
        return config
