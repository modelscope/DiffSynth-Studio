import torch
from ..core import ModelConfig
from ..models.model_loader import ModelPool

class Metric(torch.nn.Module):
    
    @staticmethod
    def tensor_to_list(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()
        return value if isinstance(value, list) else [value]

    @staticmethod
    def download_and_load_models(model_configs: list[ModelConfig], torch_dtype: torch.dtype = torch.float32, device="cuda", vram_limit: float = None):
        model_pool = ModelPool()
        for model_config in model_configs:
            model_config.download_if_necessary()
            vram_config = model_config.vram_config()
            vram_config["computation_dtype"] = vram_config["computation_dtype"] or torch_dtype or torch.float32
            vram_config["computation_device"] = vram_config["computation_device"] or device
            model_pool.auto_load_model(
                model_config.path,
                vram_config=vram_config,
                vram_limit=vram_limit,
                clear_parameters=model_config.clear_parameters,
                state_dict=model_config.state_dict,
            )
        return model_pool
