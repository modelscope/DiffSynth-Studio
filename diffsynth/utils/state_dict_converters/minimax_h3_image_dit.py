def MiniMaxH3ImageDiTComfyStateDictConverter(state_dict):
    return {key: state_dict[key] for key in state_dict if not key.endswith(".comfy_attention.config")}
