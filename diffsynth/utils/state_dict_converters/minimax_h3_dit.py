NUM_ATTENTION_HEADS = 56
# Every tensor whose rows are laid out per attention head: the packed weight, and for the
# quantized releases its per-output-channel scale.
QKV_ROW_SUFFIXES = (".attn.qkv_proj.weight", ".attn.qkv_proj.weight_scale", ".attn.qkv_proj.bias")


def MiniMaxH3DiTComfyOrgStateDictConverter(state_dict):
    """
    Regroup the fused qkv rows of a Comfy-Org MiniMax-H3 DiT checkpoint.

    Comfy-Org stores `attn.qkv_proj` as three blocks of whole heads (q | k | v), while
    `MiniMaxH3DiT` expects q, k and v interleaved per head. The key names and shapes are
    identical either way -- so the file even hashes the same as the official release -- and
    loading it unconverted raises nothing but produces flat grey video and clipped audio.
    """
    converted = {}
    for name in state_dict:
        tensor = state_dict[name]
        if name.endswith(QKV_ROW_SUFFIXES):
            tensor = _regroup_qkv_rows(tensor, name)
        converted[name] = tensor
    return converted


def _regroup_qkv_rows(tensor, name):
    rows = tensor.shape[0]
    if rows % (3 * NUM_ATTENTION_HEADS) != 0:
        raise ValueError(
            f"Cannot regroup the qkv rows of `{name}`: {rows} rows are not divisible by "
            f"3 * {NUM_ATTENTION_HEADS} heads. This converter is specific to the MiniMax-H3 "
            "DiT released by Comfy-Org."
        )
    trailing = tensor.shape[1:]
    return (tensor.reshape(3, NUM_ATTENTION_HEADS, -1, *trailing)
                  .transpose(0, 1)
                  .contiguous()
                  .reshape(rows, *trailing))
