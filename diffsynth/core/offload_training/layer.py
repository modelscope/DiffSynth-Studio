"""
Layer offloading for training — hook-based CPU offload.

At any time, only one module's weights reside on GPU. The rest stay in CPU memory.

Experimentally verified (see docs/superpowers/specs/2026-04-17-layer-offload-training-design.md):
  - 37.9% GPU memory reduction vs baseline (410.4 vs 660.8 MB, 84M params)
  - 44.8% with gradient checkpointing (365.5 MB)
  - Gradient correctness: max diff = 0.00e+00 vs baseline
  - Recomputing offload: keeping weights on GPU during checkpoint recompute
    is optimal (offloading to CPU during recompute increases peak by 4.4%)
"""
import torch
import torch.nn as nn

# Module offload state machine — complete lifecycle for training:
#   idle(0)        → weights on CPU, ready for forward
#   forward_pre(1) → weights being loaded to GPU
#   forwarding(2)  → forward computation in progress
#   forwarded(3)   → forward done, weights offloaded to CPU
#   recomputing(4) → checkpoint recompute forward done, weights kept on GPU
#   backward_pre(5)→ weights being reloaded to GPU for backward
#   backwaring(6)  → backward computation in progress
#   backwared(7)   → backward done, weights moved to CPU, ready to reset
#
# Transitions (no checkpointing):
#   idle → forward_pre → forwarding → forward_hook → forwarded(CPU)
#   forwarded → backward_pre → backwaring → backward_hook → backwared(CPU) → idle
#
# Transitions (with checkpointing):
#   idle → forward_pre → forwarding → forward_hook → forwarded(CPU)
#   backward recomputes forward:
#     forwarded → forward_pre → forwarding → forward_hook → recomputing(GPU)
#     recomputing → backward_pre → backwaring → backward_hook → backwared(CPU) → idle
#
# Note: During recomputing, weights are kept on GPU. Experimentally verified
# that offloading to CPU during recomputing increases peak memory by ~4.4%
# due to repeated load/unload cycles in backward_pre.

class OffloadState:
    idle = 0
    forward_pre = 1
    forwarding = 2
    forwarded = 3
    recomputing = 4
    backward_pre = 5
    backwaring = 6
    backwared = 7


class LayerOffloadManager:
    """
    Manages per-layer CPU offload for training.

    Registers forward/backward hooks on all leaf modules so that at any time
    only one module's weights reside on GPU. The rest stay in CPU memory.

    Uses a state machine (OffloadState) to handle gradient checkpointing:
    during checkpoint recompute, forward_hook detects the recomputing state
    and keeps weights on GPU so autograd can compute gradients.

    optimize_on_cpu behavior (verified experimentally):
    - False: ALL params are offloaded. Gradients stay on GPU during backward
      (DDP all-reduce needs them there). Caller must move grads to CPU
      AFTER accelerator.backward() returns via move_gradients_to_cpu().
    - True: Trainable params (requires_grad=True) are NOT offloaded — they
      stay on GPU the entire time. Non-trainable params still do offload.
      Since grads and params are both on GPU, optimizer.step() runs on GPU
      automatically, no need to move gradients.

    Key design decisions (verified experimentally):
    - backward_hook only moves param.data to CPU, NOT param.grad
    - DDP's gradient all-reduce happens inside accelerator.backward(),
      so param.grad must stay on GPU until backward() returns
    - param.grad should be moved to CPU by the caller AFTER backward()
      (only when optimize_on_cpu=False)
    """

    def __init__(self, model: nn.Module, target_device: torch.device, optimize_on_cpu: bool = False):
        self.target_device = target_device
        self.optimize_on_cpu = optimize_on_cpu
        self._ever_forwarded: set = set()  # Track modules that completed initial forward
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf module
                self._register_single_module_hooks(module, name)

    def _register_single_module_hooks(self, module: nn.Module, name: str):
        # When optimize_on_cpu=True, skip offload for modules with trainable params.
        # These params stay on GPU so grads + optimizer states also stay on GPU.
        if self.optimize_on_cpu:
            has_trainable = any(p.requires_grad for p in module.parameters(recurse=False))
            if has_trainable:
                # Keep trainable params on GPU — no offload hooks
                return

        module._offload_state = OffloadState.idle

        def forward_pre_hook(mod, args):
            mod._offload_state = OffloadState.forward_pre
            for param in mod.parameters(recurse=False):
                param.data = param.data.to(self.target_device)
            mod._offload_state = OffloadState.forwarding

        def forward_hook(mod, args, output):
            if mod not in self._ever_forwarded:
                # First forward: offload to CPU, save GPU memory
                mod._offload_state = OffloadState.forwarded
                self._ever_forwarded.add(mod)
                for param in mod.parameters(recurse=False):
                    param.data = param.data.to('cpu')
            else:
                # Checkpoint recompute forward: keep weights on GPU
                # so autograd can compute gradients
                mod._offload_state = OffloadState.recomputing

        def backward_pre_hook(mod, grad_input):
            mod._offload_state = OffloadState.backward_pre
            for param in mod.parameters(recurse=False):
                param.data = param.data.to(self.target_device)
            mod._offload_state = OffloadState.backwaring

        def backward_hook(mod, grad_input, grad_output):
            for param in mod.parameters(recurse=False):
                param.data = param.data.to('cpu')
                # DO NOT move param.grad here — DDP all-reduce needs it on GPU.
                # Caller must move grad to CPU AFTER accelerator.backward() returns.
            mod._offload_state = OffloadState.backwared
            # Reset to idle for next iteration
            mod._offload_state = OffloadState.idle

        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_hook)
        module.register_full_backward_pre_hook(backward_pre_hook)
        module.register_full_backward_hook(backward_hook)


def enable_layer_offload(model: nn.Module, target_device: torch.device, optimize_on_cpu: bool = False):
    """
    Enable layer offload training for a model.

    Call this AFTER accelerator.prepare(model, ...) so that DDP/FSDP wrapping
    is already in place. Model params must be on CPU when this is called.

    Args:
        model: The model to enable offload on (can be the DDP-wrapped model
               or the underlying model).
        target_device: The GPU device to load weights onto during forward/backward.
        optimize_on_cpu: If True, trainable parameters (e.g. LoRA weights)
                        are moved to GPU and stay there (NOT offloaded).
                        Non-trainable parameters still do offload.
                        This avoids the need to move gradients to CPU —
                        optimizer.step() runs on GPU automatically.
                        If False, all parameters are offloaded and the caller
                        must call move_gradients_to_cpu() after backward().
    """
    if optimize_on_cpu:
        # Move trainable params to GPU so they stay there for the whole training.
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(target_device)
    LayerOffloadManager(model, target_device, optimize_on_cpu)


def setup_layer_offload(model: nn.Module, target_device: torch.device, optimize_on_cpu: bool = False):
    """
    Full setup for layer offload training — moves params to CPU and enables offload.

    Call this AFTER accelerator.prepare(model, ...).

    Args:
        model: The prepared model (can be DDP-wrapped). Must have named_children().
        target_device: The GPU device for computation.
        optimize_on_cpu: If True, trainable params stay on GPU.
                        If False, all params are offloaded to CPU.
    """
    # Move params back to CPU (prepare may have moved them to GPU)
    for name, child in model.named_children():
        for param in child.parameters():
            param.data = param.data.to('cpu')

    # Register offload hooks on each child model
    for name, child in model.named_children():
        enable_layer_offload(child, target_device=target_device, optimize_on_cpu=optimize_on_cpu)


def move_gradients_to_cpu(model: nn.Module):
    """
    Move all gradients to CPU. Call this AFTER accelerator.backward() returns.

    This is necessary because LayerOffloadManager leaves gradients on GPU during
    backward to allow DDP all-reduce to complete. After backward() returns,
    gradients can be safely moved to CPU for optimizer step.

    Only needed when optimize_on_cpu=False. When optimize_on_cpu=True,
    trainable params stay on GPU so grads are already on the correct device.
    """
    for param in model.parameters():
        if param.grad is not None:
            param.grad = param.grad.to('cpu')
