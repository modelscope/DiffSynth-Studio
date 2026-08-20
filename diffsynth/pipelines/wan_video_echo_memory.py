from typing import Iterable, Optional

from ..core import ModelConfig, load_state_dict


DEFAULT_REPO_ID = "Echo-Team/Echo-Memory"
DEFAULT_FILENAME = "context_k1/epoch-0.safetensors"
DEFAULT_DOWNLOAD_SOURCE = "huggingface"
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


def _resolve_model_path(
    model_config: Optional[ModelConfig] = None,
    local_path: Optional[str] = None,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    download_source: str = DEFAULT_DOWNLOAD_SOURCE,
):
    if local_path is not None:
        return local_path
    if model_config is None:
        model_config = ModelConfig(model_id=repo_id, origin_file_pattern=filename, download_source=download_source)
    model_config.download_if_necessary()
    if isinstance(model_config.path, list):
        if len(model_config.path) != 1:
            raise ValueError(f"Expected exactly one Echo-Memory checkpoint, got {len(model_config.path)} files.")
        return model_config.path[0]
    return model_config.path


def _canonical_key(key: str) -> str:
    return ".".join(part for part in key.split(".") if part != "module")


def _source_key_candidates(key: str):
    yield key
    for prefix in ("pipe.dit.", "dit."):
        if key.startswith(prefix):
            yield key[len(prefix):]


def _has_meta_tensor(state_dict: dict) -> bool:
    return any(getattr(value, "is_meta", False) for value in state_dict.values())


def align_dit_state_dict_to_model(state_dict: dict, model_state_dict: dict) -> tuple[dict, list[str]]:
    target_key_map = {}
    for key in model_state_dict:
        target_key_map.setdefault(_canonical_key(key), []).append(key)
    aligned, unaligned = {}, []
    for key, value in state_dict.items():
        matches = []
        for candidate in _source_key_candidates(key):
            for target_key in target_key_map.get(_canonical_key(candidate), []):
                if getattr(model_state_dict[target_key], "shape", None) == getattr(value, "shape", None):
                    matches.append(target_key)
        matches = sorted(set(matches))
        if len(matches) == 1:
            aligned[matches[0]] = value
        else:
            unaligned.append(key)
    return aligned, unaligned


def load_echo_memory_dit(
    pipe,
    model_config: Optional[ModelConfig] = None,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    local_path: Optional[str] = None,
    download_source: str = DEFAULT_DOWNLOAD_SOURCE,
    torch_dtype=None,
):
    if getattr(pipe, "dit", None) is None:
        raise ValueError("pipe.dit is empty; load Wan 2.1 1.3B before overlaying Echo-Memory.")

    ckpt_path = _resolve_model_path(
        model_config=model_config,
        local_path=local_path,
        repo_id=repo_id,
        filename=filename,
        download_source=download_source,
    )
    raw = load_state_dict(ckpt_path, torch_dtype=torch_dtype, device="cpu")
    filtered = filter_dit_state_dict(raw)
    model_state_dict = pipe.dit.state_dict()
    aligned, unaligned = align_dit_state_dict_to_model(filtered, model_state_dict)
    missing, unexpected = pipe.dit.load_state_dict(aligned, strict=False, assign=_has_meta_tensor(model_state_dict))
    print(
        f"[Echo-Memory] overlaid {len(aligned)}/{len(raw)} DiT keys from {ckpt_path} "
        f"(skipped={len(raw) - len(filtered)}, unaligned={len(unaligned)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if unaligned:
        print("[Echo-Memory] unaligned example:", ", ".join(sorted(unaligned)[:5]))
    if missing:
        print("[Echo-Memory] missing example:", ", ".join(sorted(missing)[:5]))
    if unexpected:
        print("[Echo-Memory] unexpected example:", ", ".join(sorted(unexpected)[:5]))
    return missing, unexpected
