def WanAnimateAdapterStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("pose_patch_embedding.") or name.startswith("face_adapter") or name.startswith("face_encoder") or name.startswith("motion_encoder"):
            state_dict_[name] = state_dict[name]
    return state_dict_