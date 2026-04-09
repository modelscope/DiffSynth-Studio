def Flux2TextEncoderStateDictConverter(state_dict):
    rename_dict = {
        "multi_modal_projector.linear_1.weight": "model.multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_2.weight": "model.multi_modal_projector.linear_2.weight",
        "multi_modal_projector.norm.weight": "model.multi_modal_projector.norm.weight",
        "multi_modal_projector.patch_merger.merging_layer.weight": "model.multi_modal_projector.patch_merger.merging_layer.weight",
        "language_model.lm_head.weight": "lm_head.weight",
    }
    state_dict_ = {}
    for k in state_dict:
        k_ = k
        k_ = k_.replace("language_model.model", "model.language_model")
        k_ = k_.replace("vision_tower", "model.vision_tower")
        if k_ in rename_dict:
            k_ = rename_dict[k_]
        state_dict_[k_] = state_dict[k]
    return state_dict_
