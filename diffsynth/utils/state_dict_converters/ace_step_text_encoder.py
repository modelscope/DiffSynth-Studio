def AceStepTextEncoderStateDictConverter(state_dict):
    """
    将 ACE-Step Text Encoder 权重添加 model. 前缀

    Args:
        state_dict: 原始的 state dict（可能是 dict 或 DiskMap）

    Returns:
        转换后的 state dict，所有 key 添加 "model." 前缀
    """
    state_dict_ = {}
    # 处理 DiskMap 或普通 dict
    keys = state_dict.keys() if hasattr(state_dict, 'keys') else state_dict
    for k in keys:
        v = state_dict[k]
        if not k.startswith("model."):
            k = "model." + k
        state_dict_[k] = v
    return state_dict_
