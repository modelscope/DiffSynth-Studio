import torch

def SDXLTextEncoder2StateDictConverter(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if key == "text_projection.weight":
            val = state_dict[key]
            new_state_dict["model.text_projection.weight"] = val.float() if val.dtype == torch.float16 else val
        elif key.startswith("text_model.") and "position_ids" not in key:
            new_key = "model." + key
            val = state_dict[key]
            new_state_dict[new_key] = val.float() if val.dtype == torch.float16 else val
    return new_state_dict
