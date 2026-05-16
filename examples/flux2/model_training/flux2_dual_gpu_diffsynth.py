"""Dual-GPU model-parallel for FLUX.2 in DiffSynth-Studio.

Drop-in helper for DiffSynth-Studio
(https://github.com/modelscope/DiffSynth-Studio) ``examples/flux2/model_training/``
that splits the ``Flux2DiT`` transformer across two CUDA devices at the
``single_transformer_blocks`` midpoint. Enables FLUX.2-dev LoRA training
on pairs of 24+ GB consumer GPUs (2× RTX 3090, 2× RTX 4090, 2× RTX 5090)
— on a single 24 GB card the FLUX.2 transformer can't fit alongside
activations even with WDDM unified-memory paging.

Companion to the validated ai-toolkit patch
(https://github.com/genno-whittlery/flux2-dual-gpu-lora) and parallel
ports for musubi-tuner, OneTrainer, and HuggingFace diffusers. The
DiffSynth port is small because:

- ``Flux2DiT`` is structurally identical to the diffusers
  ``Flux2Transformer2DModel`` (same field names: ``x_embedder``,
  ``context_embedder``, ``transformer_blocks``,
  ``single_transformer_blocks``, ``norm_out``, ``proj_out``,
  ``time_guidance_embed``, ``*_modulation*``, ``pos_embed``). The
  identical helper structure transfers directly.
- DiffSynth uses PEFT (``inject_adapter_in_model`` in
  ``diffsynth.diffusion.training_module``) for LoRA injection, so
  per-layer LoRA routing follows the base layer's device automatically.
- The forward isn't overridden — pre-hooks bridge devices at the split
  point and at ``norm_out`` (the boundary back to cuda:0).

Env vars:
    DIFFSYNTH_DUAL_GPU=true                    enable dual-GPU path
    DIFFSYNTH_DUAL_GPU_SPLIT_AT=24             override split index
                                           (default: num_single // 2)

Usage in DiffSynth-Studio's training script
(``examples/flux2/model_training/train.py``)::

    from flux2_dual_gpu_diffsynth import enable_flux2_dual_gpu

    # ...build pipeline / training module normally...
    training_module = Flux2ImageTrainingModule(...)

    # Activate the split (env-gated; no-op when DIFFSYNTH_DUAL_GPU is unset).
    # Call AFTER LoRA injection (switch_pipe_to_training_mode) so PEFT
    # has wrapped target modules. The split places the wrapped modules
    # and PEFT LoRA params follow the base layer's device automatically.
    enable_flux2_dual_gpu(training_module.pipe.dit)

    # ...accelerator.prepare with device_placement=False for the
    #    training_module, since the transformer is already distributed...

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
    """True iff ``DIFFSYNTH_DUAL_GPU=true`` in the environment."""
    return os.getenv("DIFFSYNTH_DUAL_GPU", "false").lower() == "true"


def get_split_at(num_single_blocks: int) -> int:
    """Single-blocks split index. Override via ``DIFFSYNTH_DUAL_GPU_SPLIT_AT``."""
    override = os.getenv("DIFFSYNTH_DUAL_GPU_SPLIT_AT")
    if override is not None:
        return int(override)
    return num_single_blocks // 2


def enable_flux2_dual_gpu(dit: nn.Module) -> nn.Module:
    """Distribute the DiffSynth Flux2DiT across cuda:0 and cuda:1.

    When ``DIFFSYNTH_DUAL_GPU`` is unset this is a no-op pass-through.

    Call after the Flux2DiT (typically ``training_module.pipe.dit``) is
    loaded and after any PEFT LoRA injection has run. PEFT places LoRA
    params on the base layer's device automatically, so calling this
    function after LoRA injection puts everything in the right place.

    Returns the (in-place modified) dit.
    """
    if not is_dual_gpu_enabled():
        return dit

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"DIFFSYNTH_DUAL_GPU=true requires ≥2 CUDA devices, found "
            f"{torch.cuda.device_count()}."
        )

    num_single = len(dit.single_transformer_blocks)
    split_at = get_split_at(num_single)
    if not 0 < split_at < num_single:
        raise RuntimeError(
            f"DIFFSYNTH_DUAL_GPU_SPLIT_AT={split_at} out of range "
            f"(dit has {num_single} single blocks)."
        )

    cuda0 = torch.device("cuda:0")
    cuda1 = torch.device("cuda:1")

    # Place pre-blocks scaffolding + all double_blocks + first half of
    # single_blocks on cuda:0. Output layers (norm_out + proj_out) stay
    # on cuda:0 — the second pre-hook below brings the activation back
    # from cuda:1 just in time for them.
    dit.x_embedder.to(cuda0)
    dit.context_embedder.to(cuda0)
    dit.time_guidance_embed.to(cuda0)
    dit.pos_embed.to(cuda0)
    dit.double_stream_modulation_img.to(cuda0)
    dit.double_stream_modulation_txt.to(cuda0)
    dit.single_stream_modulation.to(cuda0)
    for block in dit.transformer_blocks:
        block.to(cuda0)
    for block in dit.single_transformer_blocks[:split_at]:
        block.to(cuda0)
    for block in dit.single_transformer_blocks[split_at:]:
        block.to(cuda1)
    dit.norm_out.to(cuda0)
    dit.proj_out.to(cuda0)

    # Per-block hook on every cuda:1 single_block — the forward loop passes
    # loop-level constants (temb_mod_params, image_rotary_emb,
    # joint_attention_kwargs) to each block; a boundary-only hook bridges
    # only the first block's inputs, so subsequent blocks receive the
    # originals from cuda:0 and crash with a device-mismatch error.
    for block in dit.single_transformer_blocks[split_at:]:
        block.register_forward_pre_hook(
            _make_device_bridge_hook(cuda1), with_kwargs=True
        )
    dit.norm_out.register_forward_pre_hook(
        _make_device_bridge_hook(cuda0), with_kwargs=True
    )

    dit._flux2_dual_gpu_split_at = split_at
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
