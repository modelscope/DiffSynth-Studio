import torch  # noqa: F401


def MiniMaxH3AudioVAEStateDictConverter(state_dict):
    """Map the checkpoint's legacy weight_norm names (weight_g / weight_v) onto
    the keys produced by torch.nn.utils.parametrizations.weight_norm
    (parametrizations.weight.original0 / original1). All other keys pass through
    unchanged (1:1 with the checkpoint).

    original0 == weight_g (magnitude), original1 == weight_v (direction),
    per _WeightNorm.right_inverse which returns (weight_g, weight_v).
    """
    state_dict_ = {}
    for key in state_dict:
        value = state_dict[key]
        if key.endswith(".weight_g"):
            new_key = key[: -len(".weight_g")] + ".parametrizations.weight.original0"
        elif key.endswith(".weight_v"):
            new_key = key[: -len(".weight_v")] + ".parametrizations.weight.original1"
        else:
            new_key = key
        state_dict_[new_key] = value
    return state_dict_
