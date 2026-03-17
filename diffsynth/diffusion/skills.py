import torch, os, importlib, warnings, json
from typing import Dict, List, Tuple, Union
from ..core import ModelConfig, load_model
from ..core.device.npu_compatible_device import get_device_type


SkillCache = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


class SkillModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def process_inputs(self, pipe=None, **kwargs):
        return {}
    
    def forward(self, **kwargs) -> SkillCache:
        raise NotImplementedError()


class MultiSkillModel(SkillModel):
    def __init__(self, models: List[SkillModel]):
        super().__init__()
        if not isinstance(models, list):
            models = [models]
        self.models = torch.nn.ModuleList(models)

    def merge(self, kv_cache_list: List[SkillCache]) -> SkillCache:
        names = {}
        for kv_cache in kv_cache_list:
            for name in kv_cache:
                names[name] = None
        kv_cache_merged = {}
        for name in names:
            kv_list = [kv_cache.get(name) for kv_cache in kv_cache_list]
            kv_list = [kv for kv in kv_list if kv is not None]
            if len(kv_list) > 0:
                k = torch.concat([kv[0] for kv in kv_list], dim=1)
                v = torch.concat([kv[1] for kv in kv_list], dim=1)
                kv_cache_merged[name] = (k, v)
        return kv_cache_merged
    
    @torch.no_grad()
    def process_inputs(self, pipe=None, inputs: List[Dict] = None, **kwargs):
        return [(i["model_id"], self.models[i["model_id"]].process_inputs(pipe=pipe, **i)) for i in inputs]
    
    def forward(self, inputs: List[Tuple[int, Dict]], **kwargs) -> SkillCache:
        kv_cache_list = []
        for model_id, model_inputs in inputs:
            kv_cache = self.models[model_id](**model_inputs)
            kv_cache_list.append(kv_cache)
        return self.merge(kv_cache_list)


def load_skill_model(path, torch_dtype=torch.bfloat16, device="cuda", verbose=1):
    spec = importlib.util.spec_from_file_location("skill_model", os.path.join(path, "model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = load_model(
        model_class=getattr(module, 'SKILL_MODEL'),
        config=getattr(module, 'SKILL_MODEL_CONFIG') if hasattr(module, 'SKILL_MODEL_CONFIG') else None,
        path=os.path.join(path, getattr(module, 'SKILL_MODEL_PATH')),
        torch_dtype=torch_dtype,
        device=device,
    )
    if verbose > 0:
        metadata = {
            "model_architecture": getattr(module, 'SKILL_MODEL').__name__,
            "code_path": os.path.join(path, "model.py"),
            "weight_path": os.path.join(path, getattr(module, 'SKILL_MODEL_PATH')),
        }
        print(f"Skill model loaded: {json.dumps(metadata, indent=4)}")
    return model


def load_skill_data_processor(path):
    spec = importlib.util.spec_from_file_location("skill_model", os.path.join(path, "model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'SKILL_DATA_PROCESSOR'):
        processor = getattr(module, 'SKILL_DATA_PROCESSOR')
        return processor
    else:
        return None


class SkillsPipeline(MultiSkillModel):
    def __init__(self, models: List[SkillModel]):
        super().__init__(models)

    @staticmethod
    def check_vram_config(model_config: ModelConfig):
        params = [
            model_config.offload_device, model_config.offload_dtype,
            model_config.onload_device, model_config.onload_dtype,
            model_config.preparing_device, model_config.preparing_dtype,
            model_config.computation_device, model_config.computation_dtype,
        ]
        for param in params:
            if param is not None:
                warnings.warn("SkillsPipeline doesn't support VRAM management. VRAM config will be ignored.")

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
    ):
        models = []
        for model_config in model_configs:
            SkillsPipeline.check_vram_config(model_config)
            model_config.download_if_necessary()
            model = load_skill_model(model_config.path, torch_dtype=torch_dtype, device=device)
            models.append(model)
        pipe = SkillsPipeline(models)
        return pipe
    
    def call_single_side(self, pipe = None, inputs: List[Dict] = None):
        inputs = self.process_inputs(pipe=pipe, inputs=inputs)
        skill_cache = self.forward(inputs)
        return skill_cache
        
    @torch.no_grad()
    def __call__(
        self,
        pipe = None,
        inputs: List[Dict] = None,
        positive_inputs: List[Dict] = None,
        negative_inputs: List[Dict] = None,
    ):
        shared_cache = self.call_single_side(pipe=pipe, inputs=inputs or [])
        positive_cache = self.call_single_side(pipe=pipe, inputs=positive_inputs or [])
        negative_cache = self.call_single_side(pipe=pipe, inputs=negative_inputs or [])
        positive_cache = self.merge([positive_cache, shared_cache])
        negative_cache = self.merge([negative_cache, shared_cache])
        return {"skill_cache": positive_cache, "negative_skill_cache": negative_cache}
