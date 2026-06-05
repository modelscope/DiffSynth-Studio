import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

LLM_TOKEN_INDICATOR = 3
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"

_BNB_SIBLING_SUFFIXES = (
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
)


class Fp8Linear(nn.Module):
    """Linear layer holding an e4m3 float8 weight + per-row float32 scale."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
        v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
    )


def is_bnb4bit_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(".quant_state.bitsandbytes__" in k for k in state_dict)


def swap_linears_to_fp8(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype,
    *,
    prefix: str = "",
) -> None:
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}{name}"
        if (
            isinstance(child, nn.Linear) and f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict
        ):
            setattr(
                module,
                name,
                Fp8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                ),
            )
        else:
            swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")


def load_fp8_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    *,
    assign: bool = False,
    strict: bool = True,
) -> None:
    prepared: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if v.dtype == FP8_WEIGHT_DTYPE:
            prepared[k] = v.to(device=device)
        elif k.endswith(FP8_SCALE_SUFFIX):
            prepared[k] = v.to(device=device, dtype=torch.float32)
        elif v.is_floating_point():
            prepared[k] = v.to(device=device, dtype=dtype)
        else:
            prepared[k] = v.to(device=device)

    missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
    if unexpected:
        raise RuntimeError(f"unexpected keys after fp8 load: {unexpected[:10]}")
    if missing:
        if strict:
            raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
        warnings.warn(f"missing keys after fp8 load: {missing[:10]}", stacklevel=2)

    model.to(device)


def swap_linears_to_bnb4bit(
    module: nn.Module,
    compute_dtype: torch.dtype,
    *,
    quant_type: str = "nf4",
    compress_statistics: bool = False,
) -> None:
    import bitsandbytes as bnb
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new_linear = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
            )
            setattr(module, name, new_linear)
        else:
            swap_linears_to_bnb4bit(
                child,
                compute_dtype,
                quant_type=quant_type,
                compress_statistics=compress_statistics,
            )


def load_bnb4bit_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    import bitsandbytes as bnb
    consumed: set[str] = set()
    for full_name, tensor in state_dict.items():
        if ".quant_state." in full_name or full_name.endswith(_BNB_SIBLING_SUFFIXES):
            continue
        parent_path, _, param_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        current = parent._parameters.get(param_name)
        if not isinstance(current, bnb.nn.Params4bit):
            continue
        prefix = full_name + "."
        quantized_stats = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
        consumed.add(full_name)
        consumed.update(quantized_stats.keys())
        parent._parameters[param_name] = bnb.nn.Params4bit.from_prequantized(
            data=tensor,
            quantized_stats=quantized_stats,
            requires_grad=False,
            device=device,
        )

    remaining = {k: v for k, v in state_dict.items() if k not in consumed}
    for k in list(remaining):
        if remaining[k].is_floating_point():
            remaining[k] = remaining[k].to(device=device, dtype=dtype)
        else:
            remaining[k] = remaining[k].to(device=device)

    missing, unexpected = model.load_state_dict(remaining, strict=False)
    real_missing = [m for m in missing if m not in consumed]
    if real_missing:
        raise RuntimeError(f"missing keys after quantized load: {real_missing[:10]}")
    if unexpected:
        raise RuntimeError(f"unexpected keys after quantized load: {unexpected[:10]}")

    for p in model.parameters():
        if isinstance(p, bnb.nn.Params4bit):
            continue
        if p.is_floating_point() and p.dtype != dtype:
            p.data = p.data.to(dtype=dtype)
        if p.device != device:
            p.data = p.data.to(device=device)
    for name, b in list(model.named_buffers()):
        if b.is_floating_point() and b.dtype != dtype:
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            parent.register_buffer(
                leaf,
                b.to(device=device, dtype=dtype),
                persistent=leaf not in parent._non_persistent_buffers_set,
            )
        elif b.device != device:
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            parent.register_buffer(
                leaf,
                b.to(device=device),
                persistent=leaf not in parent._non_persistent_buffers_set,
            )


_DEFAULT_TEXT_ENCODER_CONFIG = {
    "architectures": ["Qwen3VLModel"],
    "dtype": "bfloat16",
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "pad_token_id": None,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_theta": 5000000,
            "rope_type": "default",
        },
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": False,
    "transformers_version": "5.8.0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "dtype": "bfloat16",
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_vl_vision",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
    "ideogram_fp8_weight_only": True,
}


class Ideogram4TextEncoder(nn.Module):
    """Qwen3-VL-8B-Instruct wrapper that extracts hidden states from specific layers."""

    def __init__(self, config_path: str = None, **kwargs) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if config_path is None:
            config_kwargs = {k: v for k, v in _DEFAULT_TEXT_ENCODER_CONFIG.items() if k != "model_type"}
            config = AutoConfig.for_model("qwen3_vl", **config_kwargs)
        else:
            config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        self.config = config

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if is_fp8_state_dict(state_dict):
            swap_linears_to_fp8(self.model, state_dict, torch.bfloat16)
            return self.model.load_state_dict(state_dict, strict=False, assign=assign)
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hidden states from specific layers of Qwen3-VL.

        Args:
            token_ids: (B, L) token ids
            attention_mask: (B, L) attention mask
            text_position_ids: (B, L) position ids for text tokens

        Returns:
            (B, L, hidden_size * num_activation_layers) concatenated hidden states
        """
        from transformers.masking_utils import create_causal_mask

        language_model = self.model.language_model

        inputs_embeds = language_model.embed_tokens(token_ids)

        position_ids_4d = text_position_ids[None, ...].expand(4, text_position_ids.shape[0], -1)
        text_position_ids_4d = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids_4d,
        )
        position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured: dict[int, torch.Tensor] = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids_4d,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states

        selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
        stacked = torch.stack(selected, dim=0)
        stacked = torch.permute(stacked, (1, 2, 3, 0))
        batch_size, seq_len, hidden_size = stacked.shape[:3]
        stacked = stacked.reshape(batch_size, seq_len, -1)

        text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
        stacked = stacked * text_mask
        return stacked.to(torch.float32)
