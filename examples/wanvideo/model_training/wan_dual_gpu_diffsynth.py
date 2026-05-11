"""Dual-GPU model-parallel for Wan video DiT in DiffSynth-Studio.

Drop-in helper for DiffSynth-Studio
(https://github.com/modelscope/DiffSynth-Studio) ``examples/wanvideo/model_training/``
that splits the ``WanModel`` transformer across two CUDA devices at the
``blocks`` midpoint. Enables Wan 2.1 / 2.2 LoRA training on pairs of
24+ GB consumer GPUs (2× RTX 3090, 2× RTX 4090, 2× RTX 5090) — useful
for the 14B variants where the fp8-quantized weights fit on a single
32 GB card but video activations push training OOM at 480×832×49
frames + gradient checkpointing.

Companion to the FLUX.2 dual-GPU helper at
``examples/flux2/model_training/flux2_dual_gpu_diffsynth.py``. The
shape is similar but simpler — Wan has one block type
(``DiTBlock``) instead of FLUX.2's double + single stream split, so
the helper only registers per-block hooks across one boundary.

Env vars:
    WAN_DUAL_GPU=true                     enable dual-GPU path
    WAN_DUAL_GPU_SPLIT_AT=15              override split index
                                          (default: num_blocks // 2)

Usage in DiffSynth-Studio's training script
(``examples/wanvideo/model_training/train.py``)::

    from wan_dual_gpu_diffsynth import enable_wan_dual_gpu

    # ...build training module normally...
    training_module = WanTrainingModule(...)

    # Activate the split (env-gated; no-op when WAN_DUAL_GPU is unset).
    # Call AFTER LoRA injection (switch_pipe_to_training_mode) so PEFT
    # has wrapped target modules. The split places the wrapped modules
    # and PEFT LoRA params follow the base layer's device automatically.
    enable_wan_dual_gpu(training_module.pipe.dit)

Launch with ``--num_processes=1`` — this is *model* parallelism (both
GPUs cooperate on a single training step), not data parallelism (which
would spawn one process per GPU).
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


# ─── Env-gated public surface ───────────────────────────────────────────────

def is_dual_gpu_enabled() -> bool:
    """True iff ``WAN_DUAL_GPU=true`` in the environment."""
    return os.getenv("WAN_DUAL_GPU", "false").lower() == "true"


def get_split_at(num_blocks: int) -> int:
    """Block split index. Override via ``WAN_DUAL_GPU_SPLIT_AT``."""
    override = os.getenv("WAN_DUAL_GPU_SPLIT_AT")
    if override is not None:
        return int(override)
    return num_blocks // 2


def enable_wan_dual_gpu(dit: nn.Module) -> nn.Module:
    """Distribute the WanModel DiT across cuda:0 and cuda:1.

    When ``WAN_DUAL_GPU`` is unset this is a no-op pass-through.

    Call after the WanModel is loaded and after PEFT LoRA injection has
    run. PEFT places LoRA params on the base layer's device automatically,
    so calling this function after LoRA injection puts everything in the
    right place.

    Returns the (in-place modified) dit.
    """
    if not is_dual_gpu_enabled():
        return dit

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"WAN_DUAL_GPU=true requires ≥2 CUDA devices, found "
            f"{torch.cuda.device_count()}."
        )

    num_blocks = len(dit.blocks)
    split_at = get_split_at(num_blocks)
    if not 0 < split_at < num_blocks:
        raise RuntimeError(
            f"WAN_DUAL_GPU_SPLIT_AT={split_at} out of range "
            f"(dit has {num_blocks} blocks)."
        )

    cuda0 = torch.device("cuda:0")
    cuda1 = torch.device("cuda:1")

    # Place pre-block scaffolding + first half of blocks + output head on
    # cuda:0. WanModel uses self.freqs (RoPE precomputed table) as a
    # plain tensor attribute, not a registered buffer/parameter -- move
    # it explicitly so the .to(device) calls below don't miss it. Same
    # for any optional .img_emb / .ref_conv / .control_adapter modules
    # that some Wan variants add.
    dit.patch_embedding.to(cuda0)
    dit.text_embedding.to(cuda0)
    dit.time_embedding.to(cuda0)
    dit.time_projection.to(cuda0)
    for block in dit.blocks[:split_at]:
        block.to(cuda0)
    for block in dit.blocks[split_at:]:
        block.to(cuda1)
    dit.head.to(cuda0)
    if hasattr(dit, "img_emb") and dit.img_emb is not None:
        dit.img_emb.to(cuda0)
    if hasattr(dit, "ref_conv") and dit.ref_conv is not None:
        dit.ref_conv.to(cuda0)
    if hasattr(dit, "control_adapter") and dit.control_adapter is not None:
        dit.control_adapter.to(cuda0)

    # WanModel.freqs is a plain tuple of CPU tensors (not a registered
    # buffer); .to() on the module doesn't move it. Push it to cuda:0
    # since the patchify/freq-concat step happens there.
    if hasattr(dit, "freqs"):
        if isinstance(dit.freqs, (tuple, list)):
            dit.freqs = tuple(f.to(cuda0) if torch.is_tensor(f) else f for f in dit.freqs)
        elif torch.is_tensor(dit.freqs):
            dit.freqs = dit.freqs.to(cuda0)

    # Per-block hook on every cuda:1 block. WanModel.forward passes
    # loop-level constants (context, t_mod, freqs) positionally to each
    # block; a boundary-only hook bridges only the first block's inputs,
    # subsequent blocks receive the cuda:0 originals and crash with a
    # device-mismatch error.
    for block in dit.blocks[split_at:]:
        block.register_forward_pre_hook(
            _make_device_bridge_hook(cuda1), with_kwargs=True
        )
    # Bridge activations back to cuda:0 for the head + unpatchify.
    dit.head.register_forward_pre_hook(
        _make_device_bridge_hook(cuda0), with_kwargs=True
    )

    dit._wan_dual_gpu_split_at = split_at
    return dit


# ─── Internals ──────────────────────────────────────────────────────────────

def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested tuple/list/dict to ``device``.

    No-op when a tensor is already on ``device`` (``Tensor.to`` is an
    identity operation in that case).
    """
    if torch.is_tensor(obj):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _make_device_bridge_hook(target_device: torch.device):
    """Forward pre-hook that moves all tensor inputs to ``target_device``."""
    def hook(module, args, kwargs):
        return (
            _move_to_device(args, target_device),
            _move_to_device(kwargs, target_device),
        )
    return hook
