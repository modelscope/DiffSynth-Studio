import math
import re
from typing import Mapping

from torch import Tensor


class WanLoRAConverter:
    """Convert Wan DiT attention/FFN adapters without changing their deltas.

    DiffSynth adapters use unit scaling by default. Per-layer alpha/rank is
    folded into lora_B; other runtime scales can be supplied with ``scale``.
    Text-encoder, VACE-specific and fused-QKV adapters are not supported.
    """

    modules = {
        "self_attn.q": "attn1.to_q",
        "self_attn.k": "attn1.to_k",
        "self_attn.v": "attn1.to_v",
        "self_attn.o": "attn1.to_out.0",
        "cross_attn.q": "attn2.to_q",
        "cross_attn.k": "attn2.to_k",
        "cross_attn.v": "attn2.to_v",
        "cross_attn.o": "attn2.to_out.0",
        "ffn.0": "ffn.net.0.proj",
        "ffn.2": "ffn.net.2",
    }

    @classmethod
    def _convert(cls, state_dict, to_diffusers, scale):
        if not math.isfinite(scale):
            raise ValueError("LoRA scale must be finite.")
        mapping = (
            cls.modules if to_diffusers else {v: k for k, v in cls.modules.items()}
        )
        groups = {}
        for name, value in state_dict.items():
            for prefix in ("pipe.dit.", "transformer.", "dit."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            match = re.fullmatch(
                r"(blocks\.\d+)\.(.+)\.(lora_[AB])(?:\.default)?\.weight", name
            )
            alpha_match = re.fullmatch(r"(blocks\.\d+)\.(.+)\.alpha", name)
            if match:
                block, module, factor = match.groups()
                if value.ndim != 2 or not value.is_floating_point():
                    raise ValueError(
                        f"Expected a floating-point 2D LoRA factor: {name}"
                    )
            elif alpha_match:
                block, module = alpha_match.groups()
                factor = "alpha"
                if value.numel() != 1:
                    raise ValueError(f"Expected a scalar alpha: {name}")
            else:
                raise ValueError(f"Unsupported Wan LoRA key: {name}")
            if module not in mapping:
                raise ValueError(f"Unsupported Wan LoRA module: {module}")
            group = groups.setdefault((block, module), {})
            if factor in group:
                raise ValueError(f"Duplicate LoRA factor: {name}")
            group[factor] = value
        if not groups:
            raise ValueError("The adapter contains no supported Wan LoRA layers.")
        result = {}
        for (block, module), group in groups.items():
            if "lora_A" not in group or "lora_B" not in group:
                raise ValueError(f"Incomplete LoRA pair: {block}.{module}")
            a, b = group["lora_A"], group["lora_B"]
            rank = a.shape[0]
            if rank == 0 or b.shape[1] != rank:
                raise ValueError(f"Incompatible LoRA rank: {block}.{module}")
            multiplier = scale * (
                group["alpha"].item() / rank if "alpha" in group else 1.0
            )
            if not math.isfinite(multiplier):
                raise ValueError(f"LoRA scaling must be finite: {block}.{module}")
            target = (
                ("transformer." if to_diffusers else "") + block + "." + mapping[module]
            )
            result[target + ".lora_A.weight"] = a
            result[target + ".lora_B.weight"] = (
                b if multiplier == 1.0 else b * multiplier
            )
        return result

    @classmethod
    def align_to_opensource_format(
        cls, state_dict: Mapping[str, Tensor], scale: float = 1.0
    ) -> dict[str, Tensor]:
        return cls._convert(state_dict, True, scale)

    @classmethod
    def align_to_diffsynth_format(
        cls, state_dict: Mapping[str, Tensor], scale: float = 1.0
    ) -> dict[str, Tensor]:
        return cls._convert(state_dict, False, scale)
