import torch
from diffsynth.models.model_loader import ModelPool
from diffsynth.core.loader import ModelConfig


def test_model_loading(model_name,
                       model_config: ModelConfig,
                       vram_limit: float = None,
                       device="cpu",
                       torch_dtype=torch.bfloat16):
    model_pool = ModelPool()
    model_config.download_if_necessary()
    vram_config = model_config.vram_config()
    vram_config["computation_dtype"] = torch_dtype
    vram_config["computation_device"] = device
    model_pool.auto_load_model(
        model_config.path,
        vram_config=vram_config,
        vram_limit=vram_limit,
        clear_parameters=model_config.clear_parameters,
    )
    return model_pool.fetch_model(model_name)
