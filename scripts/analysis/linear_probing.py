#!/usr/bin/env python3
"""
Phase 2d: Linear Probing of FLUX DiT Activations (AC3D-inspired)

Three-phase pipeline producing the key paper figure (analogous to AC3D Figure 5):
  1. --prepare-targets:  Compute depth maps + variance + bit density
  2. --extract:          Hook DiT joint & single blocks, save global features
  3. --train:            Ridge regression probes → R² heatmap & line plots

The core question: "What does the diffusion model *know* internally when
conditioned on a single-photon measurement?"

Probing targets:
  - Bit density: does the model encode how much info the input carries?
  - Depth: does it encode 3D geometry without depth supervision?
  - Variance: can it predict its own uncertainty from a single pass?

Usage:
  python linear_probing.py --all \
      --lora_checkpoint models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors \
      --max_samples 100
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
JOINT_BLOCK_IDS_SPARSE = [0, 4, 9, 14, 18]       # 5 out of 19 joint blocks
SINGLE_BLOCK_IDS_SPARSE = [0, 9, 19, 28, 37]     # 5 out of 38 single blocks
JOINT_BLOCK_IDS_ALL = list(range(19))              # all 19 joint blocks
SINGLE_BLOCK_IDS_ALL = list(range(38))             # all 38 single blocks
CN_JOINT_BLOCK_IDS = list(range(5))                # ControlNet has 5 joint blocks
CN_SINGLE_BLOCK_IDS = list(range(10))              # ControlNet has 10 single blocks
TIMESTEP_INDICES = [0, 4, 9, 14, 19, 24, 27]  # 7 out of 28 denoising steps
HIDDEN_DIM = 3072
PATCH_H, PATCH_W = 32, 32                 # 512px input → 64 latent → 32 patches
DEFAULT_PCA_DIM = 0                       # PCA off by default; use --pca-dim N to enable
DEFAULT_RIDGE_LAMBDA = 0.1                # ridge regularization strength


def load_spad_image(path) -> Image.Image:
    img = Image.open(path)
    if img.mode == "I;16":
        arr = np.array(img, dtype=np.float32) * (255.0 / 65535.0)
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img.convert("RGB")


# ──────────────────────────────────────────────────────────────────────
# Activation Extractor — hooks into both joint and single blocks
# ──────────────────────────────────────────────────────────────────────
class ActivationExtractor:
    """Register forward hooks on transformer blocks to capture image-token features.

    Works with both DiT and ControlNet (same block types: FluxJointTransformerBlock,
    FluxSingleTransformerBlock). Use prefix to namespace activations (e.g., "cn_").

    Joint blocks output (img_tokens, txt_tokens) — we take img_tokens.
    Single blocks output (concat_tokens, prompt_emb) — we slice off text.
    """

    def __init__(self, model, joint_ids, single_ids, timestep_ids, prefix=""):
        self.activations = {}
        self._hooks = []
        self._current_step = None
        self._target_steps = set(timestep_ids)
        self._prefix = prefix

        for bid in joint_ids:
            if bid < len(model.blocks):
                h = model.blocks[bid].register_forward_hook(self._joint_hook(bid))
                self._hooks.append(h)

        for bid in single_ids:
            if bid < len(model.single_blocks):
                h = model.single_blocks[bid].register_forward_hook(self._single_hook(bid))
                self._hooks.append(h)

    def _joint_hook(self, bid):
        def fn(module, inp, out):
            if self._current_step not in self._target_steps:
                return
            # out = (image_tokens [B, img_len, D], text_tokens [B, txt_len, D])
            self.activations[f"{self._prefix}joint_{bid}_t{self._current_step}"] = (
                out[0].detach().float().cpu()
            )
        return fn

    def _single_hook(self, bid):
        def fn(module, inp, out):
            if self._current_step not in self._target_steps:
                return
            # out = (concat [B, txt+img, D], prompt_emb [B, txt, D])
            txt_len = out[1].shape[1]
            img = out[0][:, txt_len:].detach().float().cpu()
            self.activations[f"{self._prefix}single_{bid}_t{self._current_step}"] = img
        return fn

    def set_step(self, step_id):
        self._current_step = step_id

    def global_features(self):
        """Mean-pool over image tokens → {key: [D]}."""
        return {k: v.mean(dim=1).squeeze(0) for k, v in self.activations.items()}

    def spatial_features(self):
        """Per-token features → {key: [img_len, D]}."""
        return {k: v.squeeze(0) for k, v in self.activations.items()}

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ──────────────────────────────────────────────────────────────────────
# Streaming Spatial Prober — accumulates XTX/XTy during extraction
# ──────────────────────────────────────────────────────────────────────
class SpatialStreamingProber:
    """Ridge regression on per-token features without saving spatial files.

    Instead of writing ~420 MB/image (318 GB for 776 images), accumulates
    the sufficient statistics (XᵀX, Xᵀy, sums) in float64 during the
    extraction forward passes and solves closed-form ridge after the last
    training sample.

    Memory cost: ~75.5 MB per (block, timestep) key in float64.
    For 70 keys (10 sparse blocks × 7 timesteps): ~5.3 GB total.
    """

    def __init__(self, targets, n_train, n_total, ridge_lambda=DEFAULT_RIDGE_LAMBDA):
        """
        Args:
            targets: dict {name: tensor [n_total, H*W]} of spatial targets
            n_train: number of training samples (first n_train images)
            n_total: total number of samples
            ridge_lambda: ridge regularization strength
        """
        self.targets = targets
        self.n_train = n_train
        self.n_total = n_total
        self.ridge_lambda = ridge_lambda
        self.keys = None  # set on first accumulate call
        self.D = None
        self._accum = None
        self._weights = None
        self._intercepts = None
        self._eval_stats = None
        self._solved = False

    def _init_accumulators(self, keys, D):
        self.keys = keys
        self.D = D
        self._accum = {}
        for k in keys:
            self._accum[k] = {
                "XTX": torch.zeros(D, D, dtype=torch.float64),
                "sum_x": torch.zeros(D, dtype=torch.float64),
                "cnt": 0,
            }
            for tn in self.targets:
                self._accum[k][f"XTy_{tn}"] = torch.zeros(D, dtype=torch.float64)
                self._accum[k][f"sum_y_{tn}"] = 0.0

    def accumulate_train(self, idx, spatial_feats):
        """Accumulate statistics from one training image."""
        if self._accum is None:
            keys = sorted(spatial_feats.keys())
            D = next(iter(spatial_feats.values())).shape[-1]
            self._init_accumulators(keys, D)

        for k in self.keys:
            if k not in spatial_feats:
                continue
            x = spatial_feats[k].double()  # [n_tokens, D]
            a = self._accum[k]
            a["XTX"].add_(x.T @ x)
            a["sum_x"].add_(x.sum(0))
            a["cnt"] += x.shape[0]
            for tn, y_full in self.targets.items():
                y_i = y_full[idx].reshape(-1).double()  # [n_tokens]
                a[f"XTy_{tn}"].add_(x.T @ y_i)
                a[f"sum_y_{tn}"] += y_i.sum().item()

    def solve(self):
        """Solve ridge regression for all keys and targets."""
        assert self._accum is not None, "No training data accumulated"
        D = self.D
        self._weights = {}
        self._intercepts = {}

        for k in self.keys:
            a = self._accum[k]
            N = a["cnt"]
            if N == 0:
                continue
            mu_x = a["sum_x"] / N
            XTX_c = a["XTX"] - N * mu_x.unsqueeze(1) @ mu_x.unsqueeze(0)
            lam_s = self.ridge_lambda * XTX_c.trace() / D
            A = XTX_c + lam_s * torch.eye(D, dtype=torch.float64)

            self._weights[k] = {}
            self._intercepts[k] = {}
            for tn in self.targets:
                mu_y = a[f"sum_y_{tn}"] / N
                XTy_c = a[f"XTy_{tn}"] - N * mu_x * mu_y
                w = torch.linalg.solve(A, XTy_c.unsqueeze(1))
                b = mu_y - (mu_x @ w).item()
                self._weights[k][tn] = w.float()
                self._intercepts[k][tn] = float(b)

        self._solved = True
        del self._accum
        self._accum = None

        # Init eval accumulators
        self._eval_stats = {}
        for k in self.keys:
            if k not in self._weights:
                continue
            self._eval_stats[k] = {}
            for tn in self.targets:
                self._eval_stats[k][tn] = {
                    "ss_res": 0.0, "sy": 0.0, "sy2": 0.0,
                    "sp": 0.0, "sp2": 0.0, "syp": 0.0, "cnt": 0,
                }

    def evaluate_test(self, idx, spatial_feats):
        """Evaluate predictions on one test image."""
        assert self._solved, "Must call solve() before evaluate_test()"
        for k in self.keys:
            if k not in spatial_feats or k not in self._weights:
                continue
            x = spatial_feats[k].float()  # [n_tokens, D]
            for tn, y_full in self.targets.items():
                y_i = y_full[idx].reshape(-1).float()
                yp = (x @ self._weights[k][tn]).squeeze() + self._intercepts[k][tn]
                s = self._eval_stats[k][tn]
                s["ss_res"] += ((y_i - yp) ** 2).sum().item()
                s["sy"] += y_i.sum().item()
                s["sy2"] += (y_i ** 2).sum().item()
                s["sp"] += yp.sum().item()
                s["sp2"] += (yp ** 2).sum().item()
                s["syp"] += (y_i * yp).sum().item()
                s["cnt"] += y_i.shape[0]

    def get_results(self):
        """Compute R² and Pearson r from accumulated evaluation stats."""
        results = {}
        for tn in self.targets:
            rkey = f"spatial_{tn}"
            results[rkey] = {}
            for k in self.keys:
                if k not in self._eval_stats:
                    continue
                s = self._eval_stats[k][tn]
                if s["cnt"] == 0:
                    continue
                my = s["sy"] / s["cnt"]
                ss_tot = s["sy2"] - s["cnt"] * my * my
                r2 = 1.0 - s["ss_res"] / max(ss_tot, 1e-8)
                mp = s["sp"] / s["cnt"]
                cov = s["syp"] / s["cnt"] - my * mp
                std_y = max((s["sy2"] / s["cnt"] - my * my) ** 0.5, 1e-8)
                std_p = max((s["sp2"] / s["cnt"] - mp * mp) ** 0.5, 1e-8)
                pr = cov / (std_y * std_p)
                results[rkey][k] = {"r2": float(r2), "pearson_r": float(pr)}
        return results


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1: Target Preparation
# ══════════════════════════════════════════════════════════════════════
def prepare_targets(args):
    csv_path, dbase = Path(args.metadata_csv), Path(args.dataset_base)
    with open(csv_path) as f:
        samples = list(csv.DictReader(f))
    if args.max_samples:
        samples = samples[: args.max_samples]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "bit_density": [],
        "depth": [],
        "variance": [],
        "spatial_bit_density": [],
        "spatial_depth": [],
        "spatial_variance": [],
    }

    # ── Bit density ──────────────────────────────────────────────────
    print("Computing bit density targets …")
    for s in tqdm(samples, desc="bit density"):
        ck = "controlnet_image" if "controlnet_image" in s else "input_image"
        arr = np.array(Image.open(str(dbase / s[ck])).convert("L"), dtype=np.float32)
        if arr.max() > 1:
            arr /= 255.0
        targets["bit_density"].append(float(arr.mean()))
        spatial = np.array(
            Image.fromarray((arr * 255).astype(np.uint8)).resize(
                (PATCH_W, PATCH_H), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        targets["spatial_bit_density"].append(spatial.tolist())

    # ── Depth (DPT-Hybrid via transformers) ──────────────────────────
    print("Computing depth targets …")
    try:
        from transformers import DPTForDepthEstimation, DPTImageProcessor

        dm = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").cuda().eval()
        dp = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        for s in tqdm(samples, desc="depth"):
            gk = "target_image" if "target_image" in s else "image"
            gt = Image.open(str(dbase / s[gk])).convert("RGB")
            inp = dp(images=gt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                d = dm(**inp).predicted_depth.squeeze().cpu().numpy()
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            targets["depth"].append(float(d.mean()))
            spatial_d = np.array(
                Image.fromarray((d * 255).astype(np.uint8)).resize(
                    (PATCH_W, PATCH_H), Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
            targets["spatial_depth"].append(spatial_d.tolist())
        del dm, dp
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n  *** WARNING: Depth estimation failed ({e}) ***")
        print(f"  *** Falling back to luminance proxy — depth probe results will be unreliable! ***\n")
        for s in tqdm(samples, desc="luminance (fallback)"):
            gk = "target_image" if "target_image" in s else "image"
            gt = np.array(
                Image.open(str(dbase / s[gk])).convert("L"), dtype=np.float32
            )
            gt /= 255.0
            targets["depth"].append(float(gt.mean()))
            spatial_d = np.array(
                Image.fromarray((gt * 255).astype(np.uint8)).resize(
                    (PATCH_W, PATCH_H), Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
            targets["spatial_depth"].append(spatial_d.tolist())
        targets["_depth_fallback"] = True  # flag for downstream awareness

    # ── Variance from multi-seed outputs ─────────────────────────────
    ms = Path(args.multiseed_dir)
    seed_dirs = sorted([d for d in ms.iterdir() if d.is_dir() and d.name.startswith("seed_")]) if ms.exists() else []
    if len(seed_dirs) >= 2:
        print(f"Computing variance from {len(seed_dirs)} seeds …")
        for idx in tqdm(range(len(samples)), desc="variance"):
            fname = f"output_{idx:04d}.png"
            arrs = []
            for sd in seed_dirs:
                fp = sd / "output" / fname
                if fp.exists():
                    arrs.append(np.array(Image.open(str(fp)).convert("RGB"), dtype=np.float32) / 255.0)
            if len(arrs) >= 2:
                stacked = np.stack(arrs, axis=0)
                var_map = stacked.var(axis=0).mean(axis=-1)  # [H, W]
                targets["variance"].append(float(var_map.mean()))
                spatial_v = np.array(
                    Image.fromarray(
                        (np.clip(var_map / (var_map.max() + 1e-8), 0, 1) * 255).astype(np.uint8)
                    ).resize((PATCH_W, PATCH_H), Image.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
                targets["spatial_variance"].append(spatial_v.tolist())
            else:
                targets["variance"].append(0.0)
                targets["spatial_variance"].append(np.zeros((PATCH_H, PATCH_W)).tolist())
    else:
        print("  No multi-seed data; skipping variance targets.")
        targets["variance"] = [0.0] * len(samples)
        targets["spatial_variance"] = [np.zeros((PATCH_H, PATCH_W)).tolist()] * len(samples)

    tf = out_dir / "targets.json"
    with open(tf, "w") as f:
        json.dump(targets, f)

    for k in ["bit_density", "depth", "variance"]:
        vals = targets[k]
        nz = [v for v in vals if v > 0]
        rng = f"[{min(nz):.4f}, {max(nz):.4f}]" if nz else "all zero"
        print(f"  {k:>15s}: {rng}")
    print(f"Targets → {tf}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2: Activation Extraction
# ══════════════════════════════════════════════════════════════════════
def extract_activations(args):
    from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
    from diffsynth.utils.controlnet import ControlNetInput
    from diffsynth.utils.lora.flux import FluxLoRALoader
    from diffsynth.core import load_state_dict

    do_cn = getattr(args, 'hook_controlnet', False)
    do_spatial = getattr(args, 'spatial_streaming', False)
    no_cn = getattr(args, 'no_controlnet', False)

    print("Loading FLUX pipeline …")
    vc = dict(
        offload_dtype=torch.float8_e4m3fn, offload_device="cpu",
        onload_dtype=torch.float8_e4m3fn, onload_device="cpu",
        preparing_dtype=torch.float8_e4m3fn, preparing_device="cuda",
        computation_dtype=torch.bfloat16, computation_device="cuda",
    )
    vram = torch.cuda.mem_get_info()[1] / (1024**3) - 0.5
    mc = [
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", **vc),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vc),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vc),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vc),
        ModelConfig(model_id="InstantX/FLUX.1-dev-Controlnet-Union-alpha",
                    origin_file_pattern="diffusion_pytorch_model.safetensors"),
    ]
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=mc, vram_limit=vram,
    )

    if args.lora_checkpoint:
        print(f"Loading LoRA: {args.lora_checkpoint}")
        sd = load_state_dict(args.lora_checkpoint, torch_dtype=pipe.torch_dtype, device=pipe.device)
        FluxLoRALoader(torch_dtype=pipe.torch_dtype, device=pipe.device).fuse_lora_to_base_model(
            pipe.controlnet, sd, alpha=1.0
        )

    if getattr(args, 'all_blocks', False):
        joint_ids = JOINT_BLOCK_IDS_ALL
        single_ids = SINGLE_BLOCK_IDS_ALL
        print(f"All-blocks mode: hooking {len(joint_ids)} joint + {len(single_ids)} single DiT blocks")
    else:
        joint_ids = JOINT_BLOCK_IDS_SPARSE
        single_ids = SINGLE_BLOCK_IDS_SPARSE

    extractor = ActivationExtractor(pipe.dit, joint_ids, single_ids, TIMESTEP_INDICES)

    # ── ControlNet extractor (optional) ───────────────────────────────
    cn_extractor = None
    cn_act_dir = None
    if do_cn and not no_cn:
        # pipe.controlnet is MultiControlNet; unwrap to get the actual FluxControlNet
        cn_model = pipe.controlnet.models[0] if hasattr(pipe.controlnet, 'models') else pipe.controlnet
        cn_extractor = ActivationExtractor(
            cn_model, CN_JOINT_BLOCK_IDS, CN_SINGLE_BLOCK_IDS,
            TIMESTEP_INDICES, prefix="cn_",
        )
        cn_act_dir = Path(args.output_dir) / "cn_activations"
        cn_act_dir.mkdir(parents=True, exist_ok=True)
        print(f"ControlNet probing: hooking {len(CN_JOINT_BLOCK_IDS)} joint + "
              f"{len(CN_SINGLE_BLOCK_IDS)} single CN blocks")

    csv_path, dbase = Path(args.metadata_csv), Path(args.dataset_base)
    with open(csv_path) as f:
        samples = list(csv.DictReader(f))
    if args.max_samples:
        samples = samples[: args.max_samples]
    n = len(samples)
    n_train = int(0.8 * n)

    # ── Streaming spatial prober (optional) ───────────────────────────
    spatial_prober = None
    if do_spatial:
        tf = Path(args.output_dir) / "targets.json"
        if not tf.exists():
            print("ERROR: --spatial-streaming requires targets.json (run --prepare-targets first)")
            return
        with open(tf) as f:
            tgt = json.load(f)
        spatial_targets = {}
        # Discover all spatial targets dynamically (spatial_bit_density, spatial_depth,
        # spatial_variance, spatial_seg_*, etc.)
        for stn, vals in tgt.items():
            if not stn.startswith("spatial_"):
                continue
            if not isinstance(vals, list) or len(vals) < n:
                continue
            if stn.startswith("spatial_seg_") or stn in ("spatial_bit_density",
                    "spatial_depth", "spatial_variance",
                    "spatial_crossframe_variance"):
                arr = np.array(vals[:n])
                if arr.max() - arr.min() > 1e-8:
                    # Strip "spatial_" prefix for internal target name
                    tn = stn[len("spatial_"):]
                    spatial_targets[tn] = torch.from_numpy(arr.reshape(n, -1)).float()
        if spatial_targets:
            ridge_lam = getattr(args, 'ridge_lambda', DEFAULT_RIDGE_LAMBDA)
            spatial_prober = SpatialStreamingProber(
                spatial_targets, n_train, n, ridge_lambda=ridge_lam,
            )
            print(f"Spatial streaming: {len(spatial_targets)} targets, "
                  f"train={n_train}, test={n - n_train}")
        else:
            print("WARNING: No valid spatial targets found, disabling spatial streaming")

    act_dir = Path(args.output_dir) / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(tqdm(samples, desc="Extracting activations")):
        gpath = act_dir / f"global_{idx:04d}.pt"

        # Determine if forward pass is needed
        need_forward = not gpath.exists()
        if cn_extractor:
            cn_gpath = cn_act_dir / f"cn_global_{idx:04d}.pt"
            if not cn_gpath.exists():
                need_forward = True
        if spatial_prober is not None:
            need_forward = True  # always need spatial features

        if not need_forward:
            continue

        ck = "controlnet_image" if "controlnet_image" in sample else "input_image"
        ctrl_img = load_spad_image(str(dbase / sample[ck]))
        extractor.clear()
        if cn_extractor:
            cn_extractor.clear()

        pipe.scheduler.set_timesteps(args.steps, denoising_strength=1.0)
        cn_inputs = [] if no_cn else [ControlNetInput(image=ctrl_img, processor_id="gray", scale=1.0)]
        inp_shared = {
            "cfg_scale": 1.0, "embedded_guidance": 3.5, "t5_sequence_length": 512,
            "input_image": None, "denoising_strength": 1.0,
            "height": args.height, "width": args.width,
            "seed": args.seed + idx, "rand_device": "cpu",
            "sigma_shift": None, "num_inference_steps": args.steps,
            "multidiffusion_prompts": (), "multidiffusion_masks": (),
            "multidiffusion_scales": (),
            "kontext_images": None, "controlnet_inputs": cn_inputs,
            "ipadapter_images": None, "ipadapter_scale": 1.0,
            "eligen_entity_prompts": None, "eligen_entity_masks": None,
            "eligen_enable_on_negative": False, "eligen_enable_inpaint": False,
            "infinityou_id_image": None, "infinityou_guidance": 1.0,
            "flex_inpaint_image": None, "flex_inpaint_mask": None,
            "flex_control_image": None, "flex_control_strength": 0.5,
            "flex_control_stop": 0.5, "value_controller_inputs": None,
            "step1x_reference_image": None, "nexus_gen_reference_image": None,
            "lora_encoder_inputs": None, "lora_encoder_scale": 1.0,
            "tea_cache_l1_thresh": None,
            "tiled": False, "tile_size": 128, "tile_stride": 64,
            "progress_bar_cmd": lambda x: x,
        }
        inp_posi = {"prompt": ""}
        inp_nega = {"negative_prompt": ""}

        for unit in pipe.units:
            inp_shared, inp_posi, inp_nega = pipe.unit_runner(
                unit, pipe, inp_shared, inp_posi, inp_nega
            )
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {n: getattr(pipe, n) for n in pipe.in_iteration_models}

        with torch.no_grad():
            for pid, ts in enumerate(pipe.scheduler.timesteps):
                extractor.set_step(pid)
                if cn_extractor:
                    cn_extractor.set_step(pid)
                ts_t = ts.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
                np_ = pipe.cfg_guided_model_fn(
                    pipe.model_fn, 1.0,
                    inp_shared, inp_posi, inp_nega,
                    **models, timestep=ts_t, progress_id=pid,
                )
                inp_shared["latents"] = pipe.step(
                    pipe.scheduler, progress_id=pid,
                    noise_pred=np_, **inp_shared,
                )

        # Save DiT global features
        if not gpath.exists():
            torch.save(extractor.global_features(), gpath)

        # Save ControlNet global features
        if cn_extractor:
            cn_gpath = cn_act_dir / f"cn_global_{idx:04d}.pt"
            if not cn_gpath.exists():
                torch.save(cn_extractor.global_features(), cn_gpath)

        # Streaming spatial probing
        if spatial_prober is not None:
            spatial_feats = extractor.spatial_features()
            if idx < n_train:
                spatial_prober.accumulate_train(idx, spatial_feats)
                if idx == n_train - 1:
                    print(f"\n  Solving spatial ridge ({len(spatial_prober.keys)} keys) …")
                    spatial_prober.solve()
                    print("  Spatial weights solved.")
            else:
                spatial_prober.evaluate_test(idx, spatial_feats)

        # Save spatial files if --save-spatial (legacy disk-based approach)
        if args.save_spatial:
            spath = act_dir / f"spatial_{idx:04d}.pt"
            if not spath.exists():
                sf = {k: v.half() for k, v in extractor.spatial_features().items()}
                torch.save(sf, spath)

    # ── Save streaming spatial results ────────────────────────────────
    if spatial_prober is not None and spatial_prober._solved:
        results = spatial_prober.get_results()
        probe_dir = Path(args.output_dir) / "probes"
        probe_dir.mkdir(parents=True, exist_ok=True)
        with open(probe_dir / "spatial_streaming_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n" + "=" * 60)
        print("SPATIAL STREAMING RESULTS")
        print("=" * 60)
        for tn, data in results.items():
            if data:
                # Handle NaN R² values from near-constant test targets
                valid = {k: v for k, v in data.items()
                         if isinstance(v.get("r2"), (int, float)) and v["r2"] == v["r2"]}
                if valid:
                    best_key = max(valid, key=lambda k: valid[k]["r2"])
                    print(f"  {tn}: best R²={valid[best_key]['r2']:.4f} at {best_key}")
                else:
                    print(f"  {tn}: all R² are NaN (near-constant test targets)")
        print(f"  Saved → {probe_dir / 'spatial_streaming_results.json'}")

    extractor.remove()
    if cn_extractor:
        cn_extractor.remove()
    print(f"Activations → {act_dir}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 3: Probe Training + Figure Generation
# ══════════════════════════════════════════════════════════════════════
def _pca_reduce(X_train, X_test, n_components):
    """PCA dimensionality reduction fitted on training set.
    AC3D reduces 4096→512; we reduce 3072→n_components.
    Caps at min(n_components, n_train) since rank(X) <= n_train."""
    n_train = X_train.shape[0]
    nc = min(n_components, n_train - 1)  # rank <= n_train; leave 1 for centering
    mu = X_train.mean(0)
    Xc = X_train - mu
    # Economy SVD: only need top nc components
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    components = Vh[:nc]  # [nc, D]
    X_train_r = Xc @ components.T
    X_test_r = (X_test - mu) @ components.T
    return X_train_r, X_test_r


def _ridge_regression(X_train, y_train, X_test, y_test, lam=None, pca_dim=0):
    """Closed-form ridge regression with optional PCA and adaptive regularization.

    Properly centers both features AND targets before fitting, then adds back
    the intercept at prediction time.  Without y-centering, predictions in the
    n << D regime can have correct correlation but wildly wrong magnitude,
    producing catastrophically negative R².

    When pca_dim > 0, applies PCA before regression.
    Lambda is scaled by trace(XTX)/D for scale-invariance.
    """
    if lam is None:
        lam = DEFAULT_RIDGE_LAMBDA

    # Optional PCA reduction (critical for n << D global probing)
    if pca_dim > 0 and X_train.shape[1] > pca_dim:
        X_train, X_test = _pca_reduce(X_train, X_test, pca_dim)

    # Center and scale features
    mu_x, sd_x = X_train.mean(0), X_train.std(0).clamp(min=1e-8)
    Xn = (X_train - mu_x) / sd_x
    Xt = (X_test - mu_x) / sd_x

    # Center targets — critical for proper intercept
    mu_y = y_train.mean()
    yn = y_train - mu_y

    D = Xn.shape[1]
    XtX = Xn.T @ Xn
    lam_scaled = lam * XtX.trace() / D
    w = torch.linalg.solve(XtX + lam_scaled * torch.eye(D), Xn.T @ yn.unsqueeze(1))

    # Predict with intercept: y_pred = X @ w + mu_y
    yp = (Xt @ w).squeeze() + mu_y

    ss_res = ((y_test - yp) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    # Guard: if test targets are near-constant, R² is undefined → return NaN
    if ss_tot < 1e-6:
        r2 = float("nan")
    else:
        r2 = float((1.0 - ss_res / ss_tot).item())

    yc = y_test - y_test.mean()
    pc = yp - yp.mean()
    denom = yc.norm() * pc.norm()
    pr = float((yc * pc).sum().item() / (denom.item() + 1e-8)) if denom > 1e-8 else 0.0

    # For binary targets: add balanced accuracy
    metrics = {"r2": r2, "pearson_r": pr}
    unique_vals = y_train.unique()
    if len(unique_vals) == 2 and set(unique_vals.tolist()).issubset({0.0, 1.0}):
        pred_binary = (yp > 0.5).float()
        pos_mask = y_test == 1.0
        neg_mask = y_test == 0.0
        tpr = pred_binary[pos_mask].mean().item() if pos_mask.sum() > 0 else 0.0
        tnr = (1 - pred_binary[neg_mask]).mean().item() if neg_mask.sum() > 0 else 0.0
        metrics["balanced_acc"] = (tpr + tnr) / 2.0
        metrics["accuracy"] = (pred_binary == y_test).float().mean().item()

    return metrics


def train_probes(args):
    out_dir = Path(args.output_dir)
    probe_dir = out_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    tf = out_dir / "targets.json"
    if not tf.exists():
        print("No targets.json — run --prepare-targets first.")
        return
    with open(tf) as f:
        tgt = json.load(f)

    gfiles = sorted((out_dir / "activations").glob("global_*.pt"))
    if not gfiles:
        print("No activation files — run --extract first.")
        return

    n = len(gfiles)
    print(f"Loading {n} activation files …")
    all_feats = {}
    for fp in tqdm(gfiles, desc="load"):
        d = torch.load(fp, map_location="cpu", weights_only=True)
        for k, v in d.items():
            all_feats.setdefault(k, []).append(v)

    keys = sorted(all_feats.keys())
    for k in keys:
        all_feats[k] = torch.stack(all_feats[k])

    n_train = int(0.8 * n)
    print(f"Probing {len(keys)} (block, timestep) pairs | train={n_train}  test={n - n_train}")

    scalar_targets = {}
    # Discover all scalar targets dynamically (bit_density, depth, variance, obj_*, etc.)
    SKIP_PREFIXES = ("spatial_", "_")
    for tn, vals in tgt.items():
        if any(tn.startswith(p) for p in SKIP_PREFIXES):
            continue
        if not isinstance(vals, list) or len(vals) < n:
            continue
        # Check values are scalar (not nested lists)
        if isinstance(vals[0], list):
            continue
        if max(vals[:n]) - min(vals[:n]) > 1e-8:
            scalar_targets[tn] = torch.tensor(vals[:n], dtype=torch.float32)

    print(f"Active targets ({len(scalar_targets)}): {list(scalar_targets.keys())}")

    pca_dim = getattr(args, 'pca_dim', DEFAULT_PCA_DIM)
    ridge_lam = getattr(args, 'ridge_lambda', DEFAULT_RIDGE_LAMBDA)
    print(f"Global probing: PCA dim={pca_dim}, ridge λ={ridge_lam}")

    results = {}
    for tn, y in scalar_targets.items():
        y_tr, y_te = y[:n_train], y[n_train:]
        if y_tr.std() < 1e-8:
            continue
        results[tn] = {}
        for k in tqdm(keys, desc=tn):
            X = all_feats[k]
            metrics = _ridge_regression(
                X[:n_train], y_tr, X[n_train:], y_te,
                lam=ridge_lam, pca_dim=pca_dim,
            )
            results[tn][k] = metrics

    # ── ControlNet probing (if CN activation files exist) ─────────────
    cn_act_dir = out_dir / "cn_activations"
    cn_files = sorted(cn_act_dir.glob("cn_global_*.pt")) if cn_act_dir.exists() else []
    if cn_files:
        print(f"\nLoading {len(cn_files)} ControlNet activation files …")
        cn_feats = {}
        for fp in tqdm(cn_files, desc="load CN"):
            d = torch.load(fp, map_location="cpu", weights_only=True)
            for k, v in d.items():
                cn_feats.setdefault(k, []).append(v)

        cn_keys = sorted(cn_feats.keys())
        for k in cn_keys:
            cn_feats[k] = torch.stack(cn_feats[k])

        n_cn = cn_feats[cn_keys[0]].shape[0]
        n_cn_train = int(0.8 * n_cn)
        print(f"ControlNet probing: {len(cn_keys)} keys | "
              f"train={n_cn_train}  test={n_cn - n_cn_train}")

        for tn, y in scalar_targets.items():
            y_cn = y[:n_cn] if len(y) >= n_cn else y
            y_tr, y_te = y_cn[:n_cn_train], y_cn[n_cn_train:]
            if y_tr.std() < 1e-8:
                continue
            results[f"cn_{tn}"] = {}
            for k in tqdm(cn_keys, desc=f"CN {tn}"):
                X = cn_feats[k]
                metrics = _ridge_regression(
                    X[:n_cn_train], y_tr, X[n_cn_train:], y_te,
                    lam=ridge_lam, pca_dim=pca_dim,
                )
                results[f"cn_{tn}"][k] = metrics

    # ── Merge spatial streaming results if they exist ─────────────────
    spatial_streaming_f = probe_dir / "spatial_streaming_results.json"
    if spatial_streaming_f.exists():
        print(f"\nMerging spatial streaming results from {spatial_streaming_f}")
        with open(spatial_streaming_f) as f:
            streaming_results = json.load(f)
        results.update(streaming_results)

    # ── Spatial probing (disk-based: bit density, depth, variance, seg, etc.) ─
    # Uses streaming approach: accumulate XTX/XTy one file at a time
    sfiles = sorted((out_dir / "activations").glob("spatial_*.pt"))
    if sfiles:
        print(f"\nSpatial probing with {len(sfiles)} files (streaming) …")
        spatial_targets = {}
        for stn, vals in tgt.items():
            if not stn.startswith("spatial_") or not isinstance(vals, list):
                continue
            if len(vals) < n:
                continue
            arr = np.array(vals[:n])
            if arr.max() - arr.min() > 1e-8:
                spatial_targets[stn.replace("spatial_", "")] = torch.from_numpy(
                    arr.reshape(n, -1)
                ).float()  # [n, n_tokens]

        if spatial_targets:
            sample_d = torch.load(sfiles[0], map_location="cpu", weights_only=True)
            spatial_keys = sorted(sample_d.keys())
            D = next(iter(sample_d.values())).shape[-1]
            del sample_d

            # Pre-pass: compute per-key feature mean and std for normalization
            print(f"  Computing feature statistics for normalization …")
            feat_stats = {k: {"sum": torch.zeros(D, dtype=torch.float64),
                              "sum2": torch.zeros(D, dtype=torch.float64),
                              "cnt": 0} for k in spatial_keys}
            for i in tqdm(range(min(n_train, len(sfiles))), desc="feat stats"):
                d = torch.load(sfiles[i], map_location="cpu", weights_only=True)
                for k in spatial_keys:
                    x = d[k].double()  # [1024, D]
                    feat_stats[k]["sum"] += x.sum(0)
                    feat_stats[k]["sum2"] += (x ** 2).sum(0)
                    feat_stats[k]["cnt"] += x.shape[0]

            feat_mu = {}
            feat_sd = {}
            for k in spatial_keys:
                cnt = feat_stats[k]["cnt"]
                mu = feat_stats[k]["sum"] / cnt
                var = feat_stats[k]["sum2"] / cnt - mu ** 2
                feat_mu[k] = mu.float()
                feat_sd[k] = var.clamp(min=1e-12).sqrt().float()
            del feat_stats

            for tn, y_spatial in spatial_targets.items():
                rkey = f"spatial_{tn}"
                results[rkey] = {}

                # Compute target mean across training tokens for y-centering
                y_train_all = y_spatial[:n_train]  # [n_train, 1024]
                mu_y_spatial = y_train_all.mean().item()

                # Pass 1: accumulate normalized XTX and XTy (with centered y)
                accum = {k: {"XTX": torch.zeros(D, D), "XTy": torch.zeros(D, 1)} for k in spatial_keys}
                for i in tqdm(range(min(n_train, len(sfiles))), desc=f"spatial {tn} accumulate"):
                    d = torch.load(sfiles[i], map_location="cpu", weights_only=True)
                    y_i = y_spatial[i].reshape(-1, 1) - mu_y_spatial  # center targets
                    for k in spatial_keys:
                        x = (d[k].float() - feat_mu[k]) / feat_sd[k]  # normalize
                        accum[k]["XTX"] += x.T @ x
                        accum[k]["XTy"] += x.T @ y_i

                # Solve ridge for each key
                weights = {}
                for k in spatial_keys:
                    xtx = accum[k]["XTX"].double()
                    xty = accum[k]["XTy"].double()
                    lam_s = ridge_lam * xtx.trace() / D
                    weights[k] = torch.linalg.solve(
                        xtx + lam_s * torch.eye(D, dtype=torch.float64), xty
                    ).float()
                del accum

                # Pass 2: evaluate on test set (apply same normalization + intercept)
                eval_stats = {k: {"ss_res": 0.0, "sy": 0.0, "sy2": 0.0, "syp": 0.0,
                                  "sp": 0.0, "sp2": 0.0, "cnt": 0} for k in spatial_keys}
                n_test_files = min(n - n_train, len(sfiles) - n_train)
                for j in tqdm(range(n_test_files), desc=f"spatial {tn} evaluate"):
                    i = n_train + j
                    d = torch.load(sfiles[i], map_location="cpu", weights_only=True)
                    y_i = y_spatial[i].reshape(-1)
                    for k in spatial_keys:
                        x = (d[k].float() - feat_mu[k]) / feat_sd[k]  # normalize
                        yp = (x @ weights[k]).squeeze() + mu_y_spatial  # add intercept
                        s = eval_stats[k]
                        s["ss_res"] += ((y_i - yp) ** 2).sum().item()
                        s["sy"] += y_i.sum().item()
                        s["sy2"] += (y_i ** 2).sum().item()
                        s["sp"] += yp.sum().item()
                        s["sp2"] += (yp ** 2).sum().item()
                        s["syp"] += (y_i * yp).sum().item()
                        s["cnt"] += y_i.shape[0]
                del weights

                for k in spatial_keys:
                    s = eval_stats[k]
                    if s["cnt"] == 0:
                        continue
                    my = s["sy"] / s["cnt"]
                    ss_tot = s["sy2"] - s["cnt"] * my * my
                    r2 = 1.0 - s["ss_res"] / max(ss_tot, 1e-8)
                    # Pearson r
                    mp = s["sp"] / s["cnt"]
                    cov = s["syp"] / s["cnt"] - my * mp
                    std_y = max((s["sy2"] / s["cnt"] - my * my) ** 0.5, 1e-8)
                    std_p = max((s["sp2"] / s["cnt"] - mp * mp) ** 0.5, 1e-8)
                    pr = cov / (std_y * std_p)
                    results[rkey][k] = {"r2": float(r2), "pearson_r": float(pr)}

    # ── Save & print ─────────────────────────────────────────────────
    with open(probe_dir / "probing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("LINEAR PROBING RESULTS")
    print("=" * 80)
    for tn, data in results.items():
        print(f"\n─── {tn.upper()} ───")
        has_bacc = any("balanced_acc" in v for v in data.values())
        if has_bacc:
            print(f"  {'key':>25s} | {'R²':>8s} | {'Pearson r':>10s} | {'Bal Acc':>8s}")
        else:
            print(f"  {'key':>25s} | {'R²':>8s} | {'Pearson r':>10s}")
        print("  " + "─" * (60 if has_bacc else 50))
        best_key, best_r2 = None, -999
        for k in sorted(data.keys()):
            r = data[k]
            r2_val = r.get("r2", float("nan"))
            pr_val = r.get("pearson_r", 0.0)
            # Handle NaN
            r2_str = f"{r2_val:>8.4f}" if r2_val == r2_val else "     NaN"
            pr_str = f"{pr_val:>10.4f}"
            if r2_val == r2_val and r2_val > best_r2:
                best_r2, best_key = r2_val, k
            if has_bacc:
                ba = r.get("balanced_acc", float("nan"))
                ba_str = f"{ba:>8.4f}" if ba == ba else "     NaN"
                print(f"  {k:>25s} | {r2_str} | {pr_str} | {ba_str}")
            else:
                print(f"  {k:>25s} | {r2_str} | {pr_str}")
        if best_key:
            extra = ""
            if has_bacc:
                ba = data[best_key].get("balanced_acc", float("nan"))
                if ba == ba:
                    extra = f"  Bal Acc={ba:.4f}"
            print(f"  ** best: {best_key}  R²={best_r2:.4f}{extra}")

    _plot_figures(results, probe_dir)
    print(f"\nFigures & results → {probe_dir}")


# ──────────────────────────────────────────────────────────────────────
# Figure Generation
# ──────────────────────────────────────────────────────────────────────
def _parse_key(key):
    parts = key.split("_")
    btype = parts[0]
    bid = int(parts[1])
    tid = int(parts[2][1:])
    return btype, bid, tid


def _plot_figures(results, probe_dir):
    # Separate DiT global, CN global, and spatial results
    global_results = {k: v for k, v in results.items()
                      if not k.startswith("spatial_") and not k.startswith("cn_")}
    cn_results = {k: v for k, v in results.items() if k.startswith("cn_")}
    spatial_results = {k: v for k, v in results.items() if k.startswith("spatial_")}

    if not global_results:
        return

    ref = next(iter(global_results.values()))
    all_keys = list(ref.keys())
    parsed = [_parse_key(k) for k in all_keys]
    joint = sorted({(bt, bi) for bt, bi, _ in parsed if bt == "joint"}, key=lambda x: x[1])
    single = sorted({(bt, bi) for bt, bi, _ in parsed if bt == "single"}, key=lambda x: x[1])
    border = joint + single
    blabels = [f"J{bi}" for _, bi in joint] + [f"S{bi}" for _, bi in single]
    tsteps = sorted({t for _, _, t in parsed})

    for tname, data in global_results.items():
        _plot_heatmap(data, border, blabels, tsteps, len(joint), tname, probe_dir)
        _plot_lines(data, border, blabels, tsteps, len(joint), tname, probe_dir)

    if len(global_results) > 1:
        _plot_comparison(results, border, blabels, tsteps, len(joint), probe_dir)

    for tname, data in spatial_results.items():
        _plot_heatmap(data, border, blabels, tsteps, len(joint), tname, probe_dir)
        _plot_lines(data, border, blabels, tsteps, len(joint), tname, probe_dir)

    # Plot CN results with CN-specific block layout
    if cn_results:
        ref_cn = next(iter(cn_results.values()))
        cn_keys = list(ref_cn.keys())
        cn_parsed = [_parse_key(k.replace("cn_", "")) for k in cn_keys]
        cn_joint = sorted({(bt, bi) for bt, bi, _ in cn_parsed if bt == "joint"}, key=lambda x: x[1])
        cn_single = sorted({(bt, bi) for bt, bi, _ in cn_parsed if bt == "single"}, key=lambda x: x[1])
        cn_border = cn_joint + cn_single
        cn_blabels = [f"CN-J{bi}" for _, bi in cn_joint] + [f"CN-S{bi}" for _, bi in cn_single]
        cn_tsteps = sorted({t for _, _, t in cn_parsed})
        for tname, data in cn_results.items():
            # Remap keys from cn_joint_0_t0 → joint_0_t0 for plotting
            remapped = {k.replace("cn_", ""): v for k, v in data.items()}
            _plot_heatmap(remapped, cn_border, cn_blabels, cn_tsteps,
                          len(cn_joint), tname, probe_dir)
            _plot_lines(remapped, cn_border, cn_blabels, cn_tsteps,
                        len(cn_joint), tname, probe_dir)


def _plot_heatmap(data, border, blabels, tsteps, n_joint, tname, outdir):
    mat = np.zeros((len(border), len(tsteps)))
    for i, (bt, bi) in enumerate(border):
        for j, t in enumerate(tsteps):
            k = f"{bt}_{bi}_t{t}"
            mat[i, j] = data.get(k, {}).get("r2", 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = max(mat.max(), 0.1)
    vmin = min(mat.min(), 0.0)
    # Use diverging colormap if there are negative values, otherwise sequential
    if vmin < -0.05:
        cmap = "RdYlGn"
        vmin = max(vmin, -1.0)  # cap at -1 for readability
    else:
        cmap = "viridis"
        vmin = 0
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(tsteps)))
    ax.set_xticklabels([f"t={t}" for t in tsteps], fontsize=9)
    ax.set_yticks(range(len(border)))
    ax.set_yticklabels(blabels, fontsize=9)
    ax.set_xlabel("Denoising Step Index", fontsize=11)
    ax.set_ylabel("DiT Block", fontsize=11)
    nice = tname.replace("_", " ").title()
    ax.set_title(f"Linear Probe R² — {nice}", fontsize=13)
    plt.colorbar(im, label="R²")
    if n_joint > 0 and n_joint < len(border):
        ax.axhline(y=n_joint - 0.5, color="white", linewidth=1.5, linestyle="--")
        ax.text(len(tsteps) - 0.3, n_joint - 0.7, "Joint↑", color="white", fontsize=8, ha="right", va="bottom")
        ax.text(len(tsteps) - 0.3, n_joint - 0.3, "Single↓", color="white", fontsize=8, ha="right", va="top")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            c = "white" if v < vmax * 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=c)
    plt.tight_layout()
    fig.savefig(outdir / f"heatmap_{tname}.png", dpi=150)
    fig.savefig(outdir / f"heatmap_{tname}.pdf")
    plt.close(fig)


def _plot_lines(data, border, blabels, tsteps, n_joint, tname, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(tsteps)))
    for j, t in enumerate(tsteps):
        vals = [data.get(f"{bt}_{bi}_t{t}", {}).get("r2", 0) for bt, bi in border]
        ax.plot(range(len(border)), vals, "-o", color=cmap[j],
                label=f"step {t}", markersize=4, linewidth=1.5)
    if 0 < n_joint < len(border):
        ax.axvline(x=n_joint - 0.5, color="gray", ls="--", alpha=0.5, label="Joint → Single")
    ax.set_xticks(range(len(border)))
    ax.set_xticklabels(blabels, rotation=45, fontsize=9)
    ax.set_xlabel("DiT Block")
    ax.set_ylabel("R²")
    nice = tname.replace("_", " ").title()
    ax.set_title(f"Linear Probe Accuracy — {nice}", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / f"lineplot_{tname}.png", dpi=150)
    fig.savefig(outdir / f"lineplot_{tname}.pdf")
    plt.close(fig)


def _plot_comparison(results, border, blabels, tsteps, n_joint, outdir):
    # Prefer spatial results for the comparison figure
    spatial_results = {k.replace("spatial_", ""): v for k, v in results.items()
                       if k.startswith("spatial_")}
    plot_data = spatial_results if spatial_results else results
    palette = {"bit_density": "#e41a1c", "depth": "#377eb8", "variance": "#4daf4a"}
    nice_names = {"bit_density": "Bit Density", "depth": "Depth", "variance": "Uncertainty"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for tname, data in plot_data.items():
        best_t = max(tsteps, key=lambda t: np.mean([
            data.get(f"{bt}_{bi}_t{t}", {}).get("r2", 0) for bt, bi in border
        ]))
        vals = [data.get(f"{bt}_{bi}_t{best_t}", {}).get("r2", 0) for bt, bi in border]
        ax.plot(range(len(border)), vals, "-o", color=palette.get(tname, "gray"),
                label=f"{nice_names.get(tname, tname)} (step {best_t})",
                markersize=5, linewidth=2)
    if 0 < n_joint < len(border):
        ax.axvline(x=n_joint - 0.5, color="gray", ls="--", alpha=0.5,
                   label="Joint → Single")
    ax.set_xticks(range(len(border)))
    ax.set_xticklabels(blabels, rotation=45, fontsize=10)
    ax.set_xlabel("DiT Block", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    subtitle = "(Spatial Per-Token Probing)" if spatial_results else "(Global Mean-Pooled)"
    ax.set_title(f"What Does the Model Know? {subtitle}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=-0.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "comparison_best_timestep.png", dpi=150)
    fig.savefig(outdir / "comparison_best_timestep.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="AC3D-inspired linear probing of FLUX DiT")
    p.add_argument("--lora_checkpoint", type=str, default=None)
    p.add_argument("--metadata_csv", type=str,
                   default="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv")
    p.add_argument("--dataset_base", type=str,
                   default="/home/jw/engsci/thesis/spad/spad_dataset")
    p.add_argument("--output-dir", type=str, default="./probing_results")
    p.add_argument("--multiseed-dir", type=str,
                   default="./validation_outputs_multiseed")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--save-spatial", action="store_true",
                   help="Save per-token spatial features (needs ~420 MB/image for sparse blocks)")
    p.add_argument("--all-blocks", action="store_true",
                   help="Probe all 19 joint + 38 single blocks (AC3D-style full coverage)")
    p.add_argument("--hook-controlnet", action="store_true",
                   help="Also hook ControlNet blocks (5 joint + 10 single) for probing")
    p.add_argument("--spatial-streaming", action="store_true",
                   help="Stream spatial probing during extraction (avoids saving spatial files)")
    p.add_argument("--no-controlnet", action="store_true",
                   help="Run without ControlNet conditioning (ablation baseline)")
    p.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM,
                   help=f"PCA dims for global probing (0=disable, default={DEFAULT_PCA_DIM})")
    p.add_argument("--ridge-lambda", type=float, default=DEFAULT_RIDGE_LAMBDA,
                   help=f"Ridge regularization strength (default={DEFAULT_RIDGE_LAMBDA})")

    g = p.add_argument_group("phases")
    g.add_argument("--prepare-targets", action="store_true")
    g.add_argument("--extract", action="store_true")
    g.add_argument("--train", action="store_true")
    g.add_argument("--all", action="store_true")

    args = p.parse_args()
    if args.all:
        args.prepare_targets = args.extract = args.train = True
    if not any([args.prepare_targets, args.extract, args.train]):
        p.print_help()
        return

    if args.prepare_targets:
        prepare_targets(args)
    if args.extract:
        extract_activations(args)
    if args.train:
        train_probes(args)


if __name__ == "__main__":
    main()
