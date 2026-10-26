from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class AttentionBackend:
    func: Callable
    supports_causal: bool = False
    supports_window: bool = False
    supports_mask: bool = False
    supports_gqa: bool = False


_ATTENTION_BACKENDS = {}


def register_attention_implementation(
    name: str,
    func: Optional[Callable] = None,
    supports_causal: bool = False,
    supports_window: bool = False,
    supports_mask: bool = False,
    supports_gqa: bool = False,
):
    key = name.strip().lower()
    _ATTENTION_BACKENDS[key] = AttentionBackend(
        func=func,
        supports_causal=supports_causal,
        supports_window=supports_window,
        supports_mask=supports_mask,
        supports_gqa=supports_gqa,
    )


def get_attention_implementation(name: str) -> AttentionBackend:
    key = name.strip().lower()
    if key not in _ATTENTION_BACKENDS:
        available = ", ".join(list_attention_implementations()) or "none"
        raise ValueError(
            f"Unknown attention implementation: {name!r}. "
            f"Available implementations: {available}. "
            f"Please check the environment variable `DIFFSYNTH_ATTENTION_IMPLEMENTATION`."
        )
    return _ATTENTION_BACKENDS[key]


def list_attention_implementations():
    return sorted(_ATTENTION_BACKENDS)
