def AceStepDiTStateDictConverter(state_dict):
    """
    Convert ACE-Step DiT state dict to add 'model.' prefix for wrapper class.

    The wrapper class has self.model = AceStepConditionGenerationModel(config),
    so all keys need to be prefixed with 'model.'
    """
    state_dict_ = {}
    keys = state_dict.keys() if hasattr(state_dict, 'keys') else state_dict
    for k in keys:
        v = state_dict[k]
        if not k.startswith("model."):
            k = "model." + k
        state_dict_[k] = v
    return state_dict_
