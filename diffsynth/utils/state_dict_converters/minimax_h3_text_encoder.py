import torch  # noqa: F401

NUM_RETAINED_LAYERS = 50


def MiniMaxH3TextEncoderStateDictConverter(state_dict):
    """Adapt a full Qwen3-VL (Qwen3VLForConditionalGeneration) HF snapshot to the
    trimmed layer-50 encoder:
      - keep `model.language_model.*` and `model.visual.*` 1:1 (the wrapper stores
        the Qwen3VLModel under `self.model`, matching the snapshot's `model.` prefix);
      - drop `lm_head.weight` (encoder never computes logits);
      - drop decoder layers >= NUM_RETAINED_LAYERS;
      - drop `model.language_model.norm.*` (replaced by Identity).
    """
    prefix = "model.language_model.layers."
    state_dict_ = {}
    for key in state_dict:
        if key == "lm_head.weight":
            continue
        if key.startswith("model.language_model.norm."):
            continue
        if key.startswith(prefix):
            layer_idx = int(key[len(prefix):].split(".", 1)[0])
            if layer_idx >= NUM_RETAINED_LAYERS:
                continue
        state_dict_[key] = state_dict[key]
    return state_dict_
