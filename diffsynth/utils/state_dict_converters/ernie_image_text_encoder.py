def ErnieImageTextEncoderStateDictConverter(state_dict):
    """
    Maps checkpoint keys from multimodal Mistral3Model format
    to text-only Ministral3Model format.

    Checkpoint keys (Mistral3Model):
        language_model.model.layers.0.input_layernorm.weight
        language_model.model.norm.weight

    Model keys (ErnieImageTextEncoder → self.model = Ministral3Model):
        model.layers.0.input_layernorm.weight
        model.norm.weight

    Mapping: language_model. → model.
    """
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("language_model.model."):
            new_key = key.replace("language_model.model.", "model.", 1)
            new_state_dict[new_key] = state_dict[key]
    return new_state_dict
