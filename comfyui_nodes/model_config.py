from .type_defs import MODEL_CONFIG, VRAM_CONFIG, QUANT_CONFIG, MODEL_CONFIG_LIST


class ModelConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_id": ("STRING", {"default": ""}),
            "origin_file_pattern": ("STRING", {"default": ""}),
        }, "optional": {
            "vram_config": (VRAM_CONFIG,),
            "quant_config": (QUANT_CONFIG,),
            "path": ("STRING", {"default": ""}),
            "download_source": (["modelscope", "huggingface"], {"default": "modelscope"}),
            "clear_parameters": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = (MODEL_CONFIG,)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/config"

    def execute(self, model_id, origin_file_pattern, vram_config=None, quant_config=None,
                path="", download_source="modelscope", clear_parameters=False):
        from diffsynth.core import ModelConfig
        kwargs = {}
        if vram_config:
            kwargs.update(vram_config)
        if quant_config is not None:
            kwargs["quantize"] = quant_config
        if path and path.strip():
            kwargs["path"] = path.strip()
        if download_source:
            kwargs["download_source"] = download_source
        if clear_parameters:
            kwargs["clear_parameters"] = True
        return (ModelConfig(model_id=model_id.strip() or None,
                            origin_file_pattern=origin_file_pattern.strip() or None, **kwargs),)


class MergeModelConfigsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_config_1": (MODEL_CONFIG,)},
                "optional": {f"model_config_{i}": (MODEL_CONFIG,) for i in range(2, 9)}}

    RETURN_TYPES = (MODEL_CONFIG_LIST,)
    RETURN_NAMES = ("model_configs",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/config"

    def execute(self, model_config_1, **kwargs):
        return ([model_config_1] + [kwargs[name] for name in sorted(kwargs) if kwargs[name] is not None],)
