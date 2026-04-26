"""
Provenance helper for SPAD calibration artifacts.

Every saved NPZ should include a small set of metadata fields:

    artifact_kind        — string ID for the artifact type
    n_scenes_used        — int, number of scenes that contributed samples
    n_scenes_requested   — int, number originally enumerated (may differ if skipped)
    n_scenes_skipped     — int
    total_samples        — int (length of flat_values for ragged LUTs, or
                          var_count.sum() for variance stats)
    version              — script version string
    build_timestamp      — ISO 8601 datetime
    accumulator_module   — name of the count accumulator (e.g. "spad_utils_fixed")
    k_short              — int
    n_gt_frames          — int
    rotate_k             — int
    hotpix_fix_enabled   — bool
    seed                 — int
    mono_bin             — string (filename used)
    images_dir           — string
    scene_list_hash      — sha256 of the joined scene IDs
    scene_list           — np.ndarray of strings (the actual scene IDs used)

This module provides loaders and a compatibility checker so downstream
analysis scripts (variance_analysis.py, sample_p_hat.py) refuse to mix
incompatible artifacts.
"""
import numpy as np
from pathlib import Path

# Required provenance fields — fail loudly if missing.
REQUIRED = (
    "artifact_kind", "n_scenes_used", "total_samples", "version",
    "build_timestamp", "accumulator_module", "k_short",
    "n_gt_frames", "rotate_k", "scene_list_hash",
)

# Fields that MUST agree between two artifacts to be combined.
# n_scenes_used and total_samples are derived from scene_list_hash + params.
COMPATIBILITY_KEYS = (
    "scene_list_hash", "k_short", "n_gt_frames", "rotate_k",
    "hotpix_fix_enabled", "accumulator_module",
)


def load(path):
    """Load an NPZ and return (npz_handle, provenance_dict).

    Raises ValueError if any REQUIRED provenance field is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    npz = np.load(path, allow_pickle=False)
    keys = list(npz.keys())
    missing = [k for k in REQUIRED if k not in keys]
    if missing:
        raise ValueError(
            f"{path.name}: missing required provenance fields: {missing}\n"
            f"  Present keys: {keys}\n"
            f"  This artifact was likely built before provenance was added "
            f"(pre-audit2). Rebuild with the post-audit build script."
        )
    prov = {k: npz[k].item() if npz[k].shape == () else npz[k]
            for k in keys if k != "flat_values" and k != "offsets"
            and k != "cell_counts" and k != "shape"
            and k != "var_sum" and k != "var_sum2" and k != "var_count"
            and k != "scene_list" and k != "scenes_skipped_with_reason"}
    return npz, prov


def assert_compatible(*provs, keys=COMPATIBILITY_KEYS):
    """Raise if any pair of provenance dicts disagree on the given keys."""
    if len(provs) < 2:
        return
    ref = provs[0]
    for k in keys:
        if k not in ref:
            continue
        for i, p in enumerate(provs[1:], start=1):
            if k not in p:
                raise ValueError(f"Provenance #{i} missing key '{k}'")
            v0 = ref[k]
            v1 = p[k]
            # Normalize numpy scalars and bytes
            if hasattr(v0, "item"): v0 = v0.item()
            if hasattr(v1, "item"): v1 = v1.item()
            if isinstance(v0, bytes): v0 = v0.decode()
            if isinstance(v1, bytes): v1 = v1.decode()
            if v0 != v1:
                raise ValueError(
                    f"Provenance mismatch on '{k}':\n"
                    f"  artifact 0: {v0!r}\n"
                    f"  artifact {i}: {v1!r}\n"
                    f"  These artifacts came from incompatible builds and "
                    f"should NOT be combined."
                )


def summarize(prov, label="artifact"):
    """Pretty one-block summary of a provenance dict."""
    lines = [f"--- {label} provenance ---"]
    for k in REQUIRED:
        if k in prov:
            v = prov[k]
            if hasattr(v, "item"): v = v.item()
            if isinstance(v, bytes): v = v.decode()
            if k == "scene_list_hash":
                v = f"{str(v)[:16]}…"
            lines.append(f"  {k:<22} {v}")
    if "hotpix_fix_enabled" in prov:
        v = prov["hotpix_fix_enabled"]
        if hasattr(v, "item"): v = v.item()
        lines.append(f"  {'hotpix_fix_enabled':<22} {v}")
    return "\n".join(lines)
