MODEL_CONFIGS = [
    {
        "model_hash": "0319a1cb19835fb510907dd3367c95ff",
        "model_name": "qwen_image_dit",
        "model_class": "diffsynth.models.qwen_image_dit.QwenImageDiT",
    },
    {
        "model_hash": "8004730443f55db63092006dd9f7110e",
        "model_name": "qwen_image_text_encoder",
        "model_class": "diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.qwen_image_text_encoder.QwenImageTextEncoderStateDictConverter",
    },
    {
        "model_hash": "ed4ea5824d55ec3107b09815e318123a",
        "model_name": "qwen_image_vae",
        "model_class": "diffsynth.models.qwen_image_vae.QwenImageVAE",
    },
    {
        "model_hash": "073bce9cf969e317e5662cd570c3e79c",
        "model_name": "qwen_image_blockwise_controlnet",
        "model_class": "diffsynth.models.qwen_image_controlnet.QwenImageBlockWiseControlNet",
    },
    {
        "model_hash": "a9e54e480a628f0b956a688a81c33bab",
        "model_name": "qwen_image_blockwise_controlnet",
        "model_class": "diffsynth.models.qwen_image_controlnet.QwenImageBlockWiseControlNet",
        "extra_kwargs": {"additional_in_dim": 4}
    },
]
