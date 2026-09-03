import dataclasses
from .type_defs import QUANT_CONFIG

METHODS = [
    "bitsandbytes_nf4", "bitsandbytes_fp4", "torchao_int8_w8a16", "torchao_int4_w4a16",
    "torchao_fp8_w8a16", "torchao_int8_w8a8", "torchao_fp8_w8a8", "torchao_int4_w4a8",
    "torchao_mxfp8_w8a8", "torchao_mxfp4_w4a4", "torchao_nvfp4_w4a4", "torchao_nvfp4_w4a16",
    "comfy_kitchen_int8_w8a8", "comfy_kitchen_fp8_w8a8",
]


def _modules(value):
    if value is None:
        return None
    result = [part.strip() for part in str(value).replace("\n", ",").split(",") if part.strip()]
    return result or None


class QuantizationConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "method": (METHODS, {"default": METHODS[0]}),
            "mode": (["dynamic", "dequant_once"], {"default": "dynamic"}),
        }, "optional": {
            "target_modules": ("STRING", {"default": "", "multiline": True}),
            "exclude_modules": ("STRING", {"default": "", "multiline": True}),
            "load_prequantized": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = (QUANT_CONFIG,)
    RETURN_NAMES = ("quant_config",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/config"

    def execute(self, method, mode, target_modules="", exclude_modules="",
                load_prequantized=False):
        from diffsynth.core.quant import QuantizeConfig
        return (QuantizeConfig(method=method, mode=mode,
                               target_modules=_modules(target_modules),
                               exclude_modules=_modules(exclude_modules),
                               load_prequantized=load_prequantized),)


class MixedQuantizeConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "quant_config_1": (QUANT_CONFIG,),
            "load_prequantized": ("BOOLEAN", {"default": False}),
        }, "optional": {
            f"quant_config_{i}": (QUANT_CONFIG,) for i in range(2, 5)
        }}

    RETURN_TYPES = (QUANT_CONFIG,)
    RETURN_NAMES = ("quant_config",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/config"

    def execute(self, quant_config_1, load_prequantized=False, **kwargs):
        from diffsynth.core.quant import MixedQuantizeConfig, QuantizeConfig
        configs = [quant_config_1] + [
            kwargs[name] for name in sorted(kwargs) if kwargs[name] is not None
        ]
        configs = [
            dataclasses.replace(c, load_prequantized=False)
            if isinstance(c, QuantizeConfig) else c
            for c in configs
        ]
        return (MixedQuantizeConfig(configs=configs,
                                    load_prequantized=load_prequantized),)
