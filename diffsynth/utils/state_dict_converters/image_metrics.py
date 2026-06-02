import torch


def ImageMetricsCLIPStateDictConverter(state_dict):
    return {
        key: state_dict[key]
        for key in state_dict
        if key not in ("text_model.embeddings.position_ids", "vision_model.embeddings.position_ids")
    }


def ImageMetricsOpenCLIPStateDictConverter(state_dict):
    converted = {}
    for key in state_dict:
        value = state_dict[key]
        if key == "logit_scale":
            converted["logit_scale"] = value
        elif key == "token_embedding.weight":
            converted["text_model.embeddings.token_embedding.weight"] = value
        elif key == "positional_embedding":
            converted["text_model.embeddings.position_embedding.weight"] = value
        elif key.startswith("ln_final."):
            converted["text_model.final_layer_norm." + key[len("ln_final.") :]] = value
        elif key == "text_projection":
            converted["text_projection.weight"] = value.T
        elif key == "visual.class_embedding":
            converted["vision_model.embeddings.class_embedding"] = value
        elif key == "visual.conv1.weight":
            converted["vision_model.embeddings.patch_embedding.weight"] = value
        elif key == "visual.positional_embedding":
            converted["vision_model.embeddings.position_embedding.weight"] = value
        elif key.startswith("visual.ln_pre."):
            converted["vision_model.pre_layrnorm." + key[len("visual.ln_pre.") :]] = value
        elif key.startswith("visual.ln_post."):
            converted["vision_model.post_layernorm." + key[len("visual.ln_post.") :]] = value
        elif key == "visual.proj":
            converted["visual_projection.weight"] = value.T
        elif key.startswith("transformer.resblocks."):
            converted.update(_convert_open_clip_resblock("text_model.encoder.layers", key[len("transformer.resblocks.") :], value))
        elif key.startswith("visual.transformer.resblocks."):
            converted.update(_convert_open_clip_resblock("vision_model.encoder.layers", key[len("visual.transformer.resblocks.") :], value))
    return converted


def ImageMetricsImageRewardStateDictConverter(state_dict):
    from diffsynth.models.image_reward import ImageRewardModel

    converted = {}
    for key in state_dict:
        value = state_dict[key]
        if key.startswith("module."):
            key = key[len("module.") :]
        new_key, new_value = ImageRewardModel.convert_key_value(key, value)
        if new_key is not None and new_key != "blip.text_encoder.embeddings.position_ids":
            converted[new_key] = new_value
    hidden_size = converted["blip.text_encoder.embeddings.word_embeddings.weight"].shape[1]
    converted["blip.itm_head.weight"] = torch.zeros((2, hidden_size), dtype=converted["blip.text_encoder.embeddings.word_embeddings.weight"].dtype)
    converted["blip.itm_head.bias"] = torch.zeros((2,), dtype=converted["blip.text_encoder.embeddings.word_embeddings.weight"].dtype)
    return converted


def ImageMetricsAestheticStateDictConverter(state_dict):
    converted = {}
    for key in state_dict:
        value = state_dict[key]
        for prefix in ("model.", "module.", "aesthetic_model.", "aesthetics_predictor.", "predictor."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key == "vision_model.embeddings.position_ids":
            continue
        converted[key] = value
    return converted


def ImageMetricsFIDStateDictConverter(state_dict):
    return {"model." + key: state_dict[key] for key in state_dict if not key.startswith("fc.")}


def ImageMetricsHPSv3StateDictConverter(state_dict):
    converted = {}
    for key in state_dict:
        value = state_dict[key]
        if key.startswith("visual."):
            key = "model.visual." + key[len("visual.") :]
        elif key.startswith("model.visual."):
            pass
        elif key.startswith("model.") and not key.startswith("model.language_model."):
            key = "model.language_model." + key[len("model.") :]
        converted[key] = value
    return converted


def ImageMetricsUnifiedRewardStateDictConverter(state_dict):
    converted = {}
    for key in state_dict:
        value = state_dict[key]
        if key == "lm_head.weight":
            key = "model.lm_head.weight"
        elif key.startswith("model.language_model."):
            key = "model.model." + key[len("model.") :]
        elif key.startswith("model.visual."):
            key = "model.model." + key[len("model.") :]
        converted[key] = value
    return converted


def _convert_open_clip_resblock(prefix, suffix, value):
    converted = {}
    parts = suffix.split(".")
    layer = parts[0]
    rest = ".".join(parts[1:])
    layer_prefix = f"{prefix}.{layer}."
    if rest == "attn.in_proj_weight":
        q, k, v = value.chunk(3, dim=0)
        converted[layer_prefix + "self_attn.q_proj.weight"] = q
        converted[layer_prefix + "self_attn.k_proj.weight"] = k
        converted[layer_prefix + "self_attn.v_proj.weight"] = v
    elif rest == "attn.in_proj_bias":
        q, k, v = value.chunk(3, dim=0)
        converted[layer_prefix + "self_attn.q_proj.bias"] = q
        converted[layer_prefix + "self_attn.k_proj.bias"] = k
        converted[layer_prefix + "self_attn.v_proj.bias"] = v
    else:
        mapping = {
            "attn.out_proj.": "self_attn.out_proj.",
            "ln_1.": "layer_norm1.",
            "ln_2.": "layer_norm2.",
            "mlp.c_fc.": "mlp.fc1.",
            "mlp.c_proj.": "mlp.fc2.",
        }
        for source, target in mapping.items():
            if rest.startswith(source):
                converted[layer_prefix + target + rest[len(source) :]] = value
                break
    return converted
