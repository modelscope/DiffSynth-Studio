"""Optional Echo-Memory loader for upstream WanVideoPipeline.

Echo-Memory fine-tunes Wan 2.1 1.3B with the original DiffSynth / Wan key
names. This helper overlays a released row onto `pipe.dit` without changing
the core pipeline.

Extra slots from the research stack are skipped:

- `action_mlp` / `self_attn_with_action` (camera-action injection)
- `block_wise_ssm` / `videossm_hybrid` (state-space memory)
- `spatial_memory_module` (spatial memory)

Those modules, plus the multi-chunk revisit protocol, stay in
https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Memory
"""

from typing import Iterable, Optional

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


DEFAULT_REPO_ID = "Echo-Team/Echo-Memory"
DEFAULT_FILENAME = "context_k1/epoch-0.safetensors"
SKIP_SUBSTRINGS = (
    "action_mlp",
    "self_attn_with_action",
    "block_wise_ssm",
    "videossm_hybrid",
    "spatial_memory_module",
)


def filter_dit_state_dict(state_dict: dict, skip_substrings: Iterable[str] = SKIP_SUBSTRINGS) -> dict:
    skip_substrings = tuple(skip_substrings)
    return {
        key: value
        for key, value in state_dict.items()
        if not any(token in key for token in skip_substrings)
    }


def load_echo_memory_dit(
    pipe,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    local_path: Optional[str] = None,
):
    """Overlay an Echo-Memory DiT fine-tune onto `pipe.dit`.

    Parameters
    ----------
    pipe:
        A loaded `WanVideoPipeline`.
    repo_id / filename:
        Hugging Face path of a released row, e.g. `context_k1/epoch-0.safetensors`.
    local_path:
        Optional local `.safetensors` file. Overrides `repo_id` / `filename`.
    """
    if getattr(pipe, "dit", None) is None:
        raise ValueError("pipe.dit is empty; load Wan 2.1 1.3B before overlaying Echo-Memory.")

    ckpt_path = local_path or hf_hub_download(repo_id=repo_id, filename=filename)
    raw = load_file(ckpt_path)
    filtered = filter_dit_state_dict(raw)
    missing, unexpected = pipe.dit.load_state_dict(filtered, strict=False)
    print(
        f"[Echo-Memory] overlaid {len(filtered)}/{len(raw)} DiT keys from {ckpt_path} "
        f"(skipped={len(raw) - len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        print("[Echo-Memory] missing example:", ", ".join(sorted(missing)[:5]))
    if unexpected:
        print("[Echo-Memory] unexpected example:", ", ".join(sorted(unexpected)[:5]))
    return missing, unexpected
