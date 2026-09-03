from .type_defs import MODEL_CONFIG, PIPE


class LoRAClearNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": (PIPE,)}}
    RETURN_TYPES = (PIPE,)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/LoRA"
    def execute(self, pipe):
        pipe.clear_lora()
        return (pipe,)


class LoRALoadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipe": (PIPE,),
            "lora_config": (MODEL_CONFIG,),
        }, "optional": {
            "module": ("STRING", {"default": "dit", "multiline": False}),
            "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
        }}
    RETURN_TYPES = (PIPE,)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/LoRA"
    def execute(self, pipe, lora_config, module="dit", alpha=1.0):
        module = pipe.get_module(pipe, module)
        if module is None:
            raise ValueError(f"Pipeline has no module at '{module}'")
        pipe.load_lora(module, lora_config, alpha=alpha)
        return (pipe,)
