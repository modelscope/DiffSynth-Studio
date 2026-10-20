def Flux2VAEStateDictConverter(state_dict):
    rename_dict = {
        "bn.running_mean": "latent_mean",
        "bn.running_var": "latent_var",
    }
    state_dict_ = {}
    for name in state_dict:
        if name == "bn.num_batches_tracked":
            continue
        state_dict_[rename_dict.get(name, name)] = state_dict[name]
    return state_dict_
