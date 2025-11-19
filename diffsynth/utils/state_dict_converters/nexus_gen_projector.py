def NexusGenMergerStateDictConverter(state_dict):
    merger_state_dict = {}
    for key in state_dict:
        if key.startswith('embedding_merger.'):
            value = state_dict[key]
            new_key = key.replace("embedding_merger.", "")
            merger_state_dict[new_key] = value
    return merger_state_dict

def NexusGenAdapterStateDictConverter(state_dict):
    adapter_state_dict = {}
    for key in state_dict:
        if key.startswith('adapter.'):
            adapter_state_dict[key] = state_dict[key]
    return adapter_state_dict