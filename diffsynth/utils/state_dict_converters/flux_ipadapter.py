def FluxIpAdapterStateDictConverter(state_dict):
    state_dict_ = {}
    
    if "ip_adapter" in state_dict and isinstance(state_dict["ip_adapter"], dict):
        for name, param in state_dict["ip_adapter"].items():
            name_ = 'ipadapter_modules.' + name
            state_dict_[name_] = param
        
        if "image_proj" in state_dict:
            for name, param in state_dict["image_proj"].items():
                name_ = "image_proj." + name
                state_dict_[name_] = param
        return state_dict_

    for key, value in state_dict.items():
        if key.startswith("image_proj."):
            state_dict_[key] = value
        elif key.startswith("ip_adapter."):
            new_key = key.replace("ip_adapter.", "ipadapter_modules.")
            state_dict_[new_key] = value
        else:
            pass
            
    return state_dict_


def SiglipStateDictConverter(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("vision_model."):
            new_state_dict[key] = state_dict[key] 
    return new_state_dict