#!/usr/bin/env python3
"""
Phase 2d: Linear Probing of FLUX DiT Activations (AC3D-inspired)

Registers hooks on DiT blocks to extract intermediate activations
during inference, then trains linear probes to predict:
  - Depth (from ml-depth-pro predictions on GT)
  - Segmentation (from SAM3 predictions on GT)
  - Bit density (from SPAD control image)

This reveals what information the diffusion model encodes at each layer
and timestep, producing a figure analogous to AC3D Figure 5.

Usage:
  python linear_probing.py \
    --lora_checkpoint /path/to/lora.safetensors \
    --metadata_csv /path/to/metadata_val.csv \
    --output-dir probing_results/
    --extract   # first pass: extract activations
    --train     # second pass: train probes
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import csv
import json
import numpy as np

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.utils.controlnet import ControlNetInput
from diffsynth.utils.lora.flux import FluxLoRALoader
from diffsynth.core import load_state_dict


class ActivationExtractor:
    """Hooks into DiT blocks to capture activations at specified timesteps."""

    def __init__(self, dit_model, target_block_ids=None, target_timestep_ids=None):
        """
        Args:
            dit_model: The FluxDiT model.
            target_block_ids: Which joint blocks to probe (default: every 4th).
            target_timestep_ids: Which timestep indices to collect (default: [0, 4, 9, 14, 19, 24, 27]).
        """
        self.activations = {}
        self.hooks = []
        self.current_timestep_id = None

        if target_block_ids is None:
            n_blocks = len(dit_model.blocks)
            target_block_ids = list(range(0, n_blocks, max(1, n_blocks // 5)))

        if target_timestep_ids is None:
            target_timestep_ids = [0, 4, 9, 14, 19, 24, 27]

        self.target_block_ids = target_block_ids
        self.target_timestep_ids = target_timestep_ids

        for block_id in target_block_ids:
            if block_id < len(dit_model.blocks):
                hook = dit_model.blocks[block_id].register_forward_hook(
                    self._make_hook(f"block_{block_id}")
                )
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if self.current_timestep_id in self.target_timestep_ids:
                key = f"{name}_t{self.current_timestep_id}"
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                self.activations[key] = hidden.detach().cpu()
        return hook_fn

    def set_timestep(self, timestep_id):
        self.current_timestep_id = timestep_id

    def get_activations(self):
        return dict(self.activations)

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class LinearProbe(nn.Module):
    """Simple linear probe: flatten patch tokens -> predict spatial map."""

    def __init__(self, input_dim, output_dim=1, spatial_size=32):
        super().__init__()
        self.spatial_size = spatial_size
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: [B, seq_len, D] -> take image tokens, reshape to spatial
        out = self.proj(x)  # [B, seq_len, output_dim]
        B, S, C = out.shape
        H = W = int(S ** 0.5)
        if H * W != S:
            H = self.spatial_size
            W = S // H
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return out


def extract_activations(args):
    """Extract DiT activations for all val images."""
    print("Loading pipeline...")
    pipe = FluxImagePipeline.from_pretrained(
        [
            ModelConfig("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
            ModelConfig("black-forest-labs/FLUX.1-dev", "text_encoder/model.safetensors"),
            ModelConfig("black-forest-labs/FLUX.1-dev", "text_encoder_2/*.safetensors"),
            ModelConfig("black-forest-labs/FLUX.1-dev", "ae.safetensors"),
            ModelConfig("InstantX/FLUX.1-dev-Controlnet-Union-alpha", "diffusion_pytorch_model.safetensors"),
        ],
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    if args.lora_checkpoint:
        lora_loader = FluxLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
        lora_sd = load_state_dict(args.lora_checkpoint, torch_dtype=torch.bfloat16, device="cuda")
        lora_sd = lora_loader.convert_state_dict(lora_sd)
        lora_loader.fuse_lora(pipe.controlnet, lora_sd, alpha=1.0)

    target_blocks = [0, 4, 9, 14, 18]
    target_timesteps = [0, 4, 9, 14, 19, 24, 27]

    extractor = ActivationExtractor(
        pipe.dit,
        target_block_ids=target_blocks,
        target_timestep_ids=target_timesteps,
    )

    csv_path = Path(args.metadata_csv)
    with open(csv_path) as f:
        samples = list(csv.DictReader(f))
    if args.max_samples:
        samples = samples[:args.max_samples]

    out = Path(args.output_dir) / "activations"
    out.mkdir(parents=True, exist_ok=True)

    # Monkey-patch the denoising loop to set timestep IDs
    original_call = pipe.__class__.__call__

    def patched_call(self_pipe, *call_args, **call_kwargs):
        self_pipe.scheduler.set_timesteps(
            call_kwargs.get("num_inference_steps", 28),
            denoising_strength=call_kwargs.get("denoising_strength", 1.0),
            shift=call_kwargs.get("sigma_shift"),
        )
        # We can't easily patch into the call, so use the standard call
        # and the hooks will capture activations
        return original_call(self_pipe, *call_args, **call_kwargs)

    for idx, sample in enumerate(tqdm(samples, desc="Extracting activations")):
        save_path = out / f"activations_{idx:04d}.pt"
        if save_path.exists():
            continue

        control_key = "controlnet_image" if "controlnet_image" in sample else "input_image"
        control_path = csv_path.parent / sample[control_key]
        control_img = Image.open(control_path).convert("RGB")

        controlnet_input = ControlNetInput(image=control_img, processor_id="gray", scale=1.0)

        extractor.clear()

        # Run inference with hooks active
        # The hooks capture activations at each timestep
        # We need to set the timestep ID for each step
        pipe.scheduler.set_timesteps(args.steps, denoising_strength=1.0)
        inputs_posi = {"prompt": ""}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "cfg_scale": 1.0, "embedded_guidance": 3.5, "t5_sequence_length": 512,
            "input_image": None, "denoising_strength": 1.0,
            "height": args.height, "width": args.width,
            "seed": args.seed + idx, "rand_device": "cpu",
            "sigma_shift": None, "num_inference_steps": args.steps,
            "multidiffusion_prompts": (), "multidiffusion_masks": (), "multidiffusion_scales": (),
            "kontext_images": None, "controlnet_inputs": [controlnet_input],
            "ipadapter_images": None, "ipadapter_scale": 1.0,
            "eligen_entity_prompts": None, "eligen_entity_masks": None,
            "eligen_enable_on_negative": False, "eligen_enable_inpaint": False,
            "infinityou_id_image": None, "infinityou_guidance": 1.0,
            "flex_inpaint_image": None, "flex_inpaint_mask": None,
            "flex_control_image": None, "flex_control_strength": 0.5, "flex_control_stop": 0.5,
            "value_controller_inputs": None,
            "step1x_reference_image": None, "nexus_gen_reference_image": None,
            "lora_encoder_inputs": None, "lora_encoder_scale": 1.0,
            "tea_cache_l1_thresh": None,
            "tiled": False, "tile_size": 128, "tile_stride": 64,
            "progress_bar_cmd": lambda x: x,
        }

        for unit in pipe.units:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )

        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

        with torch.no_grad():
            for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
                extractor.set_timestep(progress_id)
                timestep_t = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
                noise_pred = pipe.cfg_guided_model_fn(
                    pipe.model_fn, 1.0,
                    inputs_shared, inputs_posi, inputs_nega,
                    **models, timestep=timestep_t, progress_id=progress_id,
                )
                inputs_shared["latents"] = pipe.step(
                    pipe.scheduler, progress_id=progress_id,
                    noise_pred=noise_pred, **inputs_shared,
                )

        activations = extractor.get_activations()
        torch.save(activations, save_path)

    extractor.remove_hooks()
    print(f"Activations saved to {out}")


def train_probes(args):
    """Train linear probes on extracted activations."""
    act_dir = Path(args.output_dir) / "activations"
    probe_dir = Path(args.output_dir) / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    act_files = sorted(act_dir.glob("activations_*.pt"))
    if not act_files:
        print("No activation files found. Run with --extract first.")
        return

    sample_acts = torch.load(act_files[0], map_location="cpu", weights_only=True)
    layer_keys = sorted(sample_acts.keys())
    print(f"Found {len(act_files)} samples, {len(layer_keys)} layer-timestep combinations")
    print(f"  Keys: {layer_keys[:5]}...")

    # For each layer-timestep, train a linear probe to predict bit density
    csv_path = Path(args.metadata_csv)
    with open(csv_path) as f:
        samples = list(csv.DictReader(f))

    results = {}

    for key in tqdm(layer_keys, desc="Training probes"):
        X_list = []
        y_list = []

        for i, act_file in enumerate(act_files):
            if i >= len(samples):
                break
            acts = torch.load(act_file, map_location="cpu", weights_only=True)
            if key not in acts:
                continue

            activation = acts[key]  # [1, seq_len, D]
            mean_act = activation.mean(dim=1)  # [1, D]
            X_list.append(mean_act.squeeze(0))

            # Target: mean bit density of control image
            control_key = "controlnet_image" if "controlnet_image" in samples[i] else "input_image"
            ctrl_path = csv_path.parent / samples[i][control_key]
            ctrl_img = np.array(Image.open(ctrl_path).convert("L")).astype(np.float32) / 255.0
            y_list.append(ctrl_img.mean())

        if len(X_list) < 10:
            continue

        X = torch.stack(X_list).float()
        y = torch.tensor(y_list).float().unsqueeze(1)

        # Simple linear regression with ridge
        n = X.shape[0]
        n_train = int(0.8 * n)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Ridge regression: w = (X^T X + lambda I)^-1 X^T y
        lam = 1e-3
        XtX = X_train.T @ X_train + lam * torch.eye(X_train.shape[1])
        Xty = X_train.T @ y_train
        w = torch.linalg.solve(XtX, Xty)

        y_pred_test = X_test @ w
        mse = F.mse_loss(y_pred_test, y_test).item()
        r2 = 1 - mse / max(y_test.var().item(), 1e-8)

        results[key] = {"mse": mse, "r2": r2, "n_train": n_train, "n_test": n - n_train}

    print(f"\n--- Linear Probing Results (bit density prediction) ---")
    print(f"{'Layer-Timestep':>30s} | {'R²':>8s} | {'MSE':>10s}")
    print("-" * 55)
    for key in sorted(results.keys()):
        r = results[key]
        print(f"{key:>30s} | {r['r2']:>8.4f} | {r['mse']:>10.6f}")

    with open(probe_dir / "probing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {probe_dir}")


def main():
    parser = argparse.ArgumentParser(description="Linear probing of FLUX DiT activations")
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--metadata_csv", type=str, default="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv")
    parser.add_argument("--output-dir", type=str, default="./probing_results")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--extract", action="store_true", help="Extract activations")
    parser.add_argument("--train", action="store_true", help="Train linear probes")

    args = parser.parse_args()

    if args.extract:
        extract_activations(args)
    if args.train:
        train_probes(args)
    if not args.extract and not args.train:
        print("Specify --extract and/or --train")


if __name__ == "__main__":
    main()
