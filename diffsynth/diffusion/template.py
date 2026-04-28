import torch, os, importlib, warnings, json, inspect
from typing import Dict, List, Tuple, Union
from ..core import ModelConfig, load_model
from ..core.device.npu_compatible_device import get_device_type
from ..utils.lora.merge import merge_lora


KVCache = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


class TemplateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def process_inputs(self, **kwargs):
        return {}
    
    def forward(self, **kwargs):
        raise NotImplementedError()


def check_template_model_format(model):
    if not hasattr(model, "process_inputs"):
        raise NotImplementedError("`process_inputs` is not implemented in the Template model.")
    if "kwargs" not in inspect.signature(model.process_inputs).parameters:
        raise NotImplementedError("`**kwargs` is not included in `process_inputs`.")
    if not hasattr(model, "forward"):
        raise NotImplementedError("`forward` is not implemented in the Template model.")
    if "kwargs" not in inspect.signature(model.forward).parameters:
        raise NotImplementedError("`**kwargs` is not included in `forward`.")


def load_template_model(path, torch_dtype=torch.bfloat16, device="cuda", verbose=1):
    spec = importlib.util.spec_from_file_location("template_model", os.path.join(path, "model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template_model_path = getattr(module, 'TEMPLATE_MODEL_PATH') if hasattr(module, 'TEMPLATE_MODEL_PATH') else None
    if template_model_path is not None:
        # With `TEMPLATE_MODEL_PATH`, a pretrained model will be loaded.
        model = load_model(
            model_class=getattr(module, 'TEMPLATE_MODEL'),
            config=getattr(module, 'TEMPLATE_MODEL_CONFIG') if hasattr(module, 'TEMPLATE_MODEL_CONFIG') else None,
            path=os.path.join(path, getattr(module, 'TEMPLATE_MODEL_PATH')),
            torch_dtype=torch_dtype,
            device=device,
        )
    else:
        # Without `TEMPLATE_MODEL_PATH`, a randomly initialized model or a non-model module will be loaded.
        model = module.TEMPLATE_MODEL()
        if hasattr(model, "to"):
            model = model.to(dtype=torch_dtype, device=device)
        if hasattr(model, "eval"):
            model = model.eval()
    check_template_model_format(model)
    if verbose > 0:
        metadata = {
            "model_architecture": getattr(module, 'TEMPLATE_MODEL').__name__,
            "code_path": os.path.join(path, "model.py"),
            "weight_path": template_model_path,
        }
        print(f"Template model loaded: {json.dumps(metadata, indent=4)}")
    return model


def load_template_data_processor(path):
    spec = importlib.util.spec_from_file_location("template_model", os.path.join(path, "model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'TEMPLATE_DATA_PROCESSOR'):
        processor = getattr(module, 'TEMPLATE_DATA_PROCESSOR')
        return processor
    else:
        return None


class TemplatePipeline(torch.nn.Module):
    def __init__(
        self,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        lazy_loading: bool = False,
    ):
        super().__init__()
        self.torch_dtype = torch_dtype
        self.device = device
        self.model_configs = model_configs
        self.lazy_loading = lazy_loading
        if lazy_loading:
            for model_config in model_configs:
                TemplatePipeline.check_vram_config(model_config)
                model_config.download_if_necessary()
            self.models = None
        else:
            models = []
            for model_config in model_configs:
                TemplatePipeline.check_vram_config(model_config)
                model_config.download_if_necessary()
                model = load_template_model(model_config.path, torch_dtype=torch_dtype, device=device)
                models.append(model)
            self.models = torch.nn.ModuleList(models)

    def merge_kv_cache(self, kv_cache_list: List[KVCache]) -> KVCache:
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
    
    def merge_template_cache(self, template_cache_list):
        params = sorted(list(set(sum([list(template_cache.keys()) for template_cache in template_cache_list], []))))
        template_cache_merged = {}
        for param in params:
            data = [template_cache[param] for template_cache in template_cache_list if param in template_cache]
            if param == "kv_cache":
                data = self.merge_kv_cache(data)
            elif param == "lora":
                data = merge_lora(data)
            elif len(data) == 1:
                data = data[0]
            else:
                print(f"Conflict detected: `{param}` appears in the outputs of multiple Template models. Only the first one will be retained.")
                data = data[0]
            template_cache_merged[param] = data
        return template_cache_merged

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
                warnings.warn("TemplatePipeline doesn't support VRAM management. VRAM config will be ignored.")

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        lazy_loading: bool = False,
    ):
        pipe = TemplatePipeline(torch_dtype, device, model_configs, lazy_loading)
        return pipe
    
    def fetch_model(self, model_id):
        if self.lazy_loading:
            model_config = self.model_configs[model_id]
            model_config.download_if_necessary()
            model = load_template_model(model_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            model = self.models[model_id]
        return model
    
    def call_single_side(self, pipe=None, inputs: List[Dict] = None):
        model = None
        onload_model_id = -1
        template_cache = []
        for i in inputs:
            model_id = i.get("model_id", 0)
            if model_id != onload_model_id:
                model = self.fetch_model(model_id)
                onload_model_id = model_id
            cache = model.process_inputs(pipe=pipe, **i)
            cache = model.forward(pipe=pipe, **cache)
            template_cache.append(cache)
        template_cache = self.merge_template_cache(template_cache)
        return template_cache
        
    @torch.no_grad()
    def __call__(
        self,
        pipe=None,
        template_inputs: List[Dict] = None,
        negative_template_inputs: List[Dict] = None,
        **kwargs,
    ):
        template_cache = self.call_single_side(pipe=pipe, inputs=template_inputs or [])
        negative_template_cache = self.call_single_side(pipe=pipe, inputs=negative_template_inputs or [])
        required_params = list(inspect.signature(pipe.__call__).parameters.keys())
        for param in template_cache:
            if param in required_params:
                kwargs[param] = template_cache[param]
            else:
                print(f"`{param}` is not included in the inputs of `{pipe.__class__.__name__}`. This parameter will be ignored.")
        for param in negative_template_cache:
            if "negative_" + param in required_params:
                kwargs["negative_" + param] = negative_template_cache[param]
            else:
                print(f"`{'negative_' + param}` is not included in the inputs of `{pipe.__class__.__name__}`. This parameter will be ignored.")
        return pipe(**kwargs)
