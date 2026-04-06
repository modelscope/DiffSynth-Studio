#!/usr/bin/env python3
"""
Generate publication-quality thesis/presentation figures.

Figures produced:
  1.  Hero: SPAD → Reconstruction → GT (curated, large)
  2.  Pairwise comparisons (Baseline vs DPS, vs Consistency, …)
  3.  Frame-count ablation visual strip
  4.  Master grid (all methods side-by-side)
  5.  Frame ablation metric curves (PSNR, LPIPS, CFID)
  6.  Variance overlay (reconstruction + seed-variance)
  7.  DPS ablation bar charts
  8.  Depth comparison montage (GT depth vs output depth)
  9.  Segmentation comparison montage (GT SAM3 vs output SAM3)
  10. Probing summary panel (copy existing probing figures)
  11. Leakage fix visualization
  12. Baseline summary table
  13. Contribution boundary table

Usage:
    python generate_thesis_figures.py --all
    python generate_thesis_figures.py --hero --pairwise
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────────

BASE = Path("/home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD")
SPAD_DS = Path("/home/jw/engsci/thesis/spad/spad_dataset")
OUT = BASE / "thesis_figures" / "publication"
BASELINE = BASE / "validation_outputs_scene_aware" / "seed_42"
DPS = BASE / "validation_outputs_physics_ablation" / "dps_eta1.0"
CONSISTENCY = BASE / "validation_outputs_consistency" / "epoch-0"
CONSIST_DPS = BASE / "validation_outputs_consistency_dps" / "eta1.0"
PHYSICS_BASE = BASE / "validation_outputs_physics_ablation" / "baseline"
OLD_RESULTS = BASE / "validation_results_lora_on_controlnet_seed_67"

FRAME_DIRS = {
    1:    BASE / "validation_outputs_frame_ablation" / "bits",
    4:    BASE / "validation_outputs_frame_ablation" / "bits_multi_4",
    16:   BASE / "validation_outputs_frame_ablation" / "bits_multi_16",
    64:   BASE / "validation_outputs_frame_ablation" / "bits_multi_64",
    256:  BASE / "validation_outputs_frame_ablation" / "bits_multi_256",
    1000: BASE / "validation_outputs_frame_ablation" / "bits_multi_1000",
}

SEED_DIRS = [BASE / "validation_outputs_multiseed" / f"seed_{s}"
             for s in [0, 13, 23, 42, 55, 67, 77, 88, 99, 123]]

PROBING_DIR = BASE / "probing_analysis_output"

CURATED_INDICES = [264, 735, 307, 206, 45, 42, 220, 53]

# ── Scene-ID Mapping ───────────────────────────────────────────────────────

def _build_idx_to_scene():
    """Build a mapping from validation index to scene ID using the val CSV."""
    csv_path = SPAD_DS / "bits" / "metadata_val.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    mapping = {}
    for i, row in enumerate(rows):
        ctrl = row.get("controlnet_image", "")
        scene_id = ctrl.split("/")[-1].split("_RAW")[0]
        mapping[i] = scene_id
    return mapping

def _build_mono_lookup():
    """Build scene_id -> monochrome file path lookup."""
    mono_dir = SPAD_DS / "monochrome"
    if not mono_dir.exists():
        return {}
    lookup = {}
    for f in sorted(mono_dir.iterdir()):
        if f.suffix == ".png":
            scene_id = f.stem.split("_RAW")[0]
            lookup[scene_id] = f
    return lookup

IDX_TO_SCENE = _build_idx_to_scene()
MONO_LOOKUP = _build_mono_lookup()

# ── Typography & Color Palette ─────────────────────────────────────────────

_SERIF = "Liberation Serif"
_SANS  = "DejaVu Sans"
_MONO  = "DejaVu Sans Mono"

for fpath in [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
]:
    if Path(fpath).exists():
        font_manager.fontManager.addfont(fpath)

COLORS = {
    "blue":   "#1a5276",
    "orange": "#c0392b",
    "green":  "#196f3d",
    "teal":   "#117a65",
    "purple": "#6c3483",
    "grey":   "#626567",
    "gold":   "#b7950b",
    "bg":     "#fafafa",
    "grid":   "#e0e0e0",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [_SERIF, "DejaVu Serif", "Times New Roman"],
    "font.sans-serif": [_SANS],
    "font.monospace": [_MONO],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
})


# ── Helpers ────────────────────────────────────────────────────────────────

def _load(path, fallback_size=(512, 512)):
    if path.exists():
        img = Image.open(path)
        if img.mode in ("I;16", "I"):
            arr = np.array(img, dtype=np.float64)
            arr = (arr / arr.max() * 255).clip(0, 255).astype(np.uint8)
            return np.stack([arr]*3, axis=-1)
        return np.array(img.convert("RGB"))
    return np.full((*fallback_size, 3), 200, dtype=np.uint8)


def _img(d, kind, idx):
    dirs = {"input": "input", "output": "output", "gt": "ground_truth"}
    prefixes = {"input": "input", "output": "output", "gt": "gt"}
    return d / dirs[kind] / f"{prefixes[kind]}_{idx:04d}.png"


def _variance_map(idx, size=512):
    arrays = []
    for sd in SEED_DIRS:
        p = sd / "output" / f"output_{idx:04d}.png"
        if p.exists():
            arrays.append(np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0)
    if len(arrays) < 3:
        return np.zeros((size, size, 3), dtype=np.uint8)
    var = np.var(np.stack(arrays), axis=0)
    normed = np.clip(var / 0.025, 0, 1)
    cmap = plt.cm.inferno(normed)[:, :, :3]
    return (cmap * 255).astype(np.uint8)


def _clean_axes(ax, border=True):
    ax.set_xticks([])
    ax.set_yticks([])
    if border:
        for s in ax.spines.values():
            s.set_edgecolor("#d0d0d0")
            s.set_linewidth(0.4)
    else:
        for s in ax.spines.values():
            s.set_visible(False)


def _save(fig, name, generated_list, subfolder=None):
    target = OUT / subfolder if subfolder else OUT
    target.mkdir(parents=True, exist_ok=True)
    p = target / f"{name}.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    rel = f"{subfolder}/{name}" if subfolder else name
    generated_list.append(rel)
    print(f"  -> {rel}.pdf/.png")


def _suptitle(fig, text, y=0.98):
    fig.suptitle(text, fontsize=12, fontweight="bold", y=y, fontfamily="serif")


# ── Figure 1: Hero Figure ─────────────────────────────────────────────────

def make_hero(indices=None, ncols=4):
    if indices is None:
        indices = CURATED_INDICES[:ncols]
    n = len(indices)

    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7.5))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = axes[:, np.newaxis]

    row_labels = [
        ("SPAD Input\n(1 binary frame)", COLORS["blue"]),
        ("Reconstruction\n(FLUX + ControlNet)", COLORS["green"]),
        ("Ground Truth\n(Long exposure)", COLORS["grey"]),
    ]

    for col, idx in enumerate(indices):
        imgs = [
            _load(_img(BASELINE, "input", idx)),
            _load(_img(BASELINE, "output", idx)),
            _load(_img(BASELINE, "gt", idx)),
        ]
        for row, img in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(img)
            _clean_axes(ax)
            if col == 0:
                label, color = row_labels[row]
                ax.set_ylabel(label, fontsize=9.5, fontweight="bold",
                              rotation=90, labelpad=12, color=color)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    _suptitle(fig, "Single-Frame 1-Bit SPAD  →  RGB Reconstruction")
    return fig


# ── Figure 2: Pairwise Comparisons ────────────────────────────────────────

def make_pairwise(method_a_name, method_a_dir, method_b_name, method_b_dir,
                  indices=None, nrows=4):
    if indices is None:
        indices = CURATED_INDICES[:nrows]
    n = len(indices)

    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = axes[np.newaxis, :]

    header_colors = [COLORS["grey"], COLORS["blue"], COLORS["orange"], COLORS["grey"]]
    col_labels = ["SPAD Input", method_a_name, method_b_name, "Ground Truth"]

    for row, idx in enumerate(indices):
        imgs = [
            _load(_img(BASELINE, "input", idx)),
            _load(_img(method_a_dir, "output", idx)),
            _load(_img(method_b_dir, "output", idx)),
            _load(_img(BASELINE, "gt", idx)),
        ]
        for col, img in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(img)
            _clean_axes(ax)
            if row == 0:
                ax.set_title(col_labels[col], fontsize=10, fontweight="bold",
                             pad=6, color=header_colors[col])

    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    _suptitle(fig, f"{method_a_name}  vs  {method_b_name}", y=1.01)
    return fig


# ── Figure 3: Frame-Count Ablation Strip ──────────────────────────────────

def make_frame_strip(indices=None, nrows=3):
    if indices is None:
        indices = CURATED_INDICES[:nrows]
    frame_counts = [1, 4, 16, 64, 256, 1000]
    ncols = len(frame_counts) + 1  # +1 for GT

    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(2.1 * ncols, 4.0 * nrows))
    fig.patch.set_facecolor("white")

    for row_block, idx in enumerate(indices[:nrows]):
        for col, fc in enumerate(frame_counts):
            d = FRAME_DIRS.get(fc)
            if d is None:
                continue
            inp = _load(_img(d, "input", idx))
            out = _load(_img(d, "output", idx))
            ax_inp = axes[row_block * 2, col]
            ax_out = axes[row_block * 2 + 1, col]
            ax_inp.imshow(inp)
            ax_out.imshow(out)
            _clean_axes(ax_inp)
            _clean_axes(ax_out)
            if row_block == 0:
                ax_inp.set_title(f"N = {fc}", fontsize=9, fontweight="bold",
                                 pad=4, color=COLORS["blue"])
            if col == 0:
                ax_inp.set_ylabel("Input", fontsize=8, fontweight="bold", color=COLORS["grey"])
                ax_out.set_ylabel("Output", fontsize=8, fontweight="bold", color=COLORS["grey"])

        gt_ax_inp = axes[row_block * 2, ncols - 1]
        gt_ax_out = axes[row_block * 2 + 1, ncols - 1]

        # Map validation index -> scene ID -> correct monochrome file
        scene_id = IDX_TO_SCENE.get(idx, "")
        mono_path = MONO_LOOKUP.get(scene_id)
        if mono_path and mono_path.exists():
            gt_input_img = _load(mono_path)
        else:
            gt_input_img = _load(_img(FRAME_DIRS[1], "gt", idx))
        gt_output_img = _load(_img(FRAME_DIRS[1], "gt", idx))

        gt_ax_inp.imshow(gt_input_img)
        gt_ax_out.imshow(gt_output_img)
        _clean_axes(gt_ax_inp)
        _clean_axes(gt_ax_out)
        if row_block == 0:
            gt_ax_inp.set_title("GT (λ)", fontsize=9, fontweight="bold",
                                pad=4, color=COLORS["green"])

    fig.subplots_adjust(wspace=0.02, hspace=0.04)
    _suptitle(fig, "Frame-Count Ablation: Increasing Accumulated SPAD Frames", y=0.99)
    return fig


# ── Figure 4: Master Grid ─────────────────────────────────────────────────

def make_master_grid(indices=None, nrows=6):
    if indices is None:
        indices = CURATED_INDICES[:nrows]
    n = min(nrows, len(indices))

    method_dirs = [
        ("SPAD Input",    BASELINE,    "input"),
        ("Baseline",      BASELINE,    "output"),
        ("DPS (η=1.0)",   DPS,         "output"),
        ("Consistency",   CONSISTENCY, "output"),
        ("Consist.+DPS",  CONSIST_DPS, "output"),
        ("Seed Variance", None,        "variance"),
        ("Ground Truth",  BASELINE,    "gt"),
    ]
    ncols = len(method_dirs)

    fig, axes = plt.subplots(n, ncols, figsize=(2.2 * ncols, 2.25 * n))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = axes[np.newaxis, :]

    hdr_colors = [COLORS["grey"], COLORS["blue"], COLORS["teal"],
                  COLORS["purple"], COLORS["orange"], COLORS["gold"], COLORS["grey"]]

    for row, idx in enumerate(indices[:n]):
        for col, (label, d, kind) in enumerate(method_dirs):
            ax = axes[row, col]
            if kind == "variance":
                img = _variance_map(idx)
            else:
                img = _load(_img(d, kind, idx))
            ax.imshow(img)
            _clean_axes(ax)
            if row == 0:
                ax.set_title(label, fontsize=8.5, fontweight="bold",
                             pad=4, color=hdr_colors[col])

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    _suptitle(fig, "Reconstruction Comparison Across Methods", y=1.0)
    return fig


# ── Figure 5: Metric Curves ───────────────────────────────────────────────

def make_metric_curves():
    fc = [1, 4, 16, 64, 256, 1000]
    psnr  = [17.89, 17.82, 16.91, 15.47, 14.12, 13.04]
    ssim  = [0.596, 0.636, 0.651, 0.640, 0.605, 0.551]
    lpips = [0.415, 0.376, 0.359, 0.346, 0.339, 0.347]
    fid   = [66.84, 71.44, 74.85, 74.50, 70.66, 68.52]
    cfid  = [151.94, 138.05, 131.04, 120.90, 110.11, 108.11]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.patch.set_facecolor("white")

    def _curve(ax, x, y, color, marker, ylabel, title, invert=False, annot=None):
        ax.plot(x, y, marker=marker, color=color, linewidth=2.0, markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("Accumulated Frames (N)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3, linestyle="-", color=COLORS["grid"])
        ax.set_xticks(fc)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(labelsize=8)
        if invert:
            ax.invert_yaxis()
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.2f}" if yi > 1 else f"{yi:.3f}",
                        (xi, yi), textcoords="offset points",
                        xytext=(0, -14 if not invert else 10), fontsize=6.5,
                        ha="center", color=COLORS["grey"])
        if annot:
            ax.annotate(annot[0], xy=annot[1], xytext=annot[2],
                        fontsize=7.5, ha="center", fontstyle="italic",
                        arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
                        color=COLORS["grey"])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    _curve(axes[0], fc, psnr, COLORS["blue"], "o", "PSNR (dB) ↑",
           "Distortion", annot=("Trained on\nN = 1 only", (1, 17.89), (3, 16.3)))
    _curve(axes[1], fc, lpips, COLORS["orange"], "s", "LPIPS ↓",
           "Perceptual Quality", invert=True)
    _curve(axes[2], fc, cfid, COLORS["green"], "D", "CFID ↓",
           "Conditional Fidelity", invert=True)

    _suptitle(fig, "Perception–Distortion–Consistency Trade-off Across Frame Counts", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── Figure 6: Variance Overlay ────────────────────────────────────────────

def make_variance_overlay(indices=None, ncols=4):
    if indices is None:
        indices = CURATED_INDICES[:ncols]
    n = min(len(indices), ncols)

    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.2))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = axes[:, np.newaxis]

    for col in range(n):
        idx = indices[col]
        recon = _load(_img(BASELINE, "output", idx))
        var   = _variance_map(idx)
        axes[0, col].imshow(recon)
        axes[1, col].imshow(var)
        _clean_axes(axes[0, col])
        _clean_axes(axes[1, col])
        if col == 0:
            axes[0, col].set_ylabel("Reconstruction", fontsize=9, fontweight="bold",
                                     color=COLORS["blue"])
            axes[1, col].set_ylabel("Seed Variance", fontsize=9, fontweight="bold",
                                     color=COLORS["orange"])

    fig.subplots_adjust(wspace=0.02, hspace=0.04)
    _suptitle(fig, "Multi-Seed Variance Maps  (K = 10 seeds, Inferno colormap)", y=0.99)
    return fig


# ── Figure 7: DPS Ablation ────────────────────────────────────────────────

def make_dps_ablation():
    etas  = ["0\n(baseline)", "0.01", "0.05", "0.1", "0.5", "1.0"]
    psnr  = [17.89, 18.02, 18.02, 18.02, 18.03, 18.05]
    lpips = [0.4152, 0.4132, 0.4132, 0.4132, 0.4131, 0.4131]
    cfid  = [151.94, 151.66, 151.39, 151.45, 151.50, 151.35]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    fig.patch.set_facecolor("white")
    x = np.arange(len(etas))
    w = 0.55

    def _bars(ax, vals, color, ylabel, title, ylim):
        bars = ax.bar(x, vals, w, color=color, alpha=0.85, edgecolor="white", linewidth=0.8,
                      zorder=3)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(etas, fontsize=7.5)
        ax.set_xlabel("DPS Guidance η", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="-", color=COLORS["grid"], zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + (ylim[1]-ylim[0])*0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=COLORS["grey"])

    _bars(axes[0], psnr, COLORS["blue"], "PSNR (dB) ↑", "Distortion", (17.82, 18.12))
    _bars(axes[1], lpips, COLORS["orange"], "LPIPS ↓", "Perceptual Quality", (0.4125, 0.4160))
    _bars(axes[2], cfid, COLORS["green"], "CFID ↓", "Conditional Fidelity", (150.8, 152.5))

    _suptitle(fig, "Latent-Space DPS Guidance Strength (η) Ablation", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── Figure 8: Depth Comparison ─────────────────────────────────────────────

def make_depth_montage(nrows=4):
    depth_gt_dir = OLD_RESULTS / "depth_ground_truth"
    depth_out_dir = OLD_RESULTS / "depth_output"
    if not depth_gt_dir.exists() or not depth_out_dir.exists():
        print("  [SKIP] No depth data found at old results dir")
        return None

    gt_rgb_dir = OLD_RESULTS / "ground_truth"
    out_rgb_dir = OLD_RESULTS / "output"
    input_dir = OLD_RESULTS / "input"

    indices = sorted([f.stem.split("_")[1] for f in depth_gt_dir.glob("gt_*.png")])[:nrows]
    if not indices:
        return None

    fig, axes = plt.subplots(len(indices), 6, figsize=(15, 2.5 * len(indices)))
    fig.patch.set_facecolor("white")
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["SPAD Input", "GT RGB", "Output RGB",
                  "GT Depth", "Output Depth", "|Depth Diff|"]
    hdr_colors = [COLORS["grey"], COLORS["grey"], COLORS["blue"],
                  COLORS["green"], COLORS["teal"], COLORS["orange"]]

    for row, idx in enumerate(indices):
        ctrl_path = input_dir / f"input_{idx}.png"
        gt_rgb_path = gt_rgb_dir / f"gt_{idx}.png"
        out_rgb_path = out_rgb_dir / f"output_{idx}.png"
        gt_depth_path = depth_gt_dir / f"gt_{idx}.png"
        out_depth_path = depth_out_dir / f"output_{idx}.png"

        imgs = [
            _load(ctrl_path) if ctrl_path.exists() else np.zeros((512,512,3), dtype=np.uint8),
            _load(gt_rgb_path),
            _load(out_rgb_path),
            _load(gt_depth_path),
            _load(out_depth_path),
        ]

        gt_d_npy = depth_gt_dir / f"gt_{idx}.npy"
        out_d_npy = depth_out_dir / f"output_{idx}.npy"
        if gt_d_npy.exists() and out_d_npy.exists():
            gd = np.load(gt_d_npy).astype(np.float64)
            od = np.load(out_d_npy).astype(np.float64)
            mask = (od > 0) & (gd > 0) & np.isfinite(od) & np.isfinite(gd)
            if mask.sum() > 10:
                A = np.stack([od[mask], np.ones(mask.sum())], axis=1)
                res = np.linalg.lstsq(A, gd[mask], rcond=None)
                a, b = res[0]
                aligned = a * od + b
                diff = np.abs(gd - aligned)
                hi = np.percentile(diff, 98)
                if hi < 1e-6: hi = 1.0
                diff_viz = np.clip(diff / hi, 0, 1)
                diff_rgb = (plt.cm.hot(diff_viz)[:, :, :3] * 255).astype(np.uint8)
            else:
                diff_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            diff_rgb = np.zeros((gd.shape[0] if 'gd' in dir() else 512,
                                  gd.shape[1] if 'gd' in dir() else 512, 3), dtype=np.uint8)
        imgs.append(diff_rgb)

        for col, img in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(img)
            _clean_axes(ax)
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9, fontweight="bold",
                             pad=5, color=hdr_colors[col])

    fig.subplots_adjust(wspace=0.02, hspace=0.03)
    _suptitle(fig, "Depth Estimation: Ground Truth vs Reconstructed RGB", y=1.01)
    return fig


# ── Figure 9: Segmentation Comparison ──────────────────────────────────────

def make_seg_montage(nrows=4):
    sam3_dir = OLD_RESULTS / "sam3_montage"
    if not sam3_dir.exists():
        print("  [SKIP] No segmentation montages found")
        return None

    montage_files = sorted(sam3_dir.glob("montage_*.png"))[:nrows]
    if not montage_files:
        return None

    gt_rgb_dir = OLD_RESULTS / "ground_truth"
    out_rgb_dir = OLD_RESULTS / "output"
    input_dir = OLD_RESULTS / "input"

    fig, axes = plt.subplots(len(montage_files), 5, figsize=(14, 2.8 * len(montage_files)))
    fig.patch.set_facecolor("white")
    if len(montage_files) == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["SPAD Input", "GT RGB", "Generated RGB", "GT Segmentation", "Gen Segmentation"]
    hdr_colors = [COLORS["grey"], COLORS["grey"], COLORS["blue"],
                  COLORS["green"], COLORS["teal"]]

    for row, mf in enumerate(montage_files):
        idx = mf.stem.split("_")[1]

        montage_img = Image.open(mf).convert("RGB")
        w_full, h_full = montage_img.size

        ctrl_path = input_dir / f"input_{idx}.png"
        gt_path = gt_rgb_dir / f"gt_{idx}.png"
        out_path = out_rgb_dir / f"output_{idx}.png"

        ctrl = _load(ctrl_path)
        gt_rgb = _load(gt_path)
        out_rgb = _load(out_path)

        # 5 equal-width panels separated by ~20px gaps
        pw = (w_full - 4 * 20 - 30) // 5  # ~568px each
        gap = 20
        x0 = 15
        starts = [x0 + i * (pw + gap) for i in range(5)]
        # Crop segmentation panels, skip title/header area (top ~80px)
        crop_top = 80
        gt_seg = np.array(montage_img.crop((starts[3], crop_top, starts[3] + pw, h_full)))
        gen_seg = np.array(montage_img.crop((starts[4], crop_top, starts[4] + pw, h_full)))

        panels = [ctrl, gt_rgb, out_rgb, gt_seg, gen_seg]

        for col, img in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img)
            _clean_axes(ax)
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9.5, fontweight="bold",
                             pad=5, color=hdr_colors[col])

    fig.subplots_adjust(wspace=0.02, hspace=0.03)
    _suptitle(fig, "Semantic Segmentation: Ground Truth vs Reconstructed RGB", y=1.01)
    return fig


# ── Figure 10: Probing Summary Panel ──────────────────────────────────────

def make_probing_panel():
    key_figs = [
        (PROBING_DIR / "fig1_main_heatmap.png", "R² Heatmap (LoRA Model)"),
        (PROBING_DIR / "fig3_delta_heatmap.png", "LoRA Delta R²"),
        (PROBING_DIR / "fig6_global_vs_spatial.png", "Global vs Spatial Probing"),
    ]
    existing = [(p, t) for p, t in key_figs if p.exists()]
    if not existing:
        print("  [SKIP] No probing figures found")
        return None

    n = len(existing)
    fig_h = 4 if n <= 2 else 3.5
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, fig_h))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    for i, (path, title) in enumerate(existing):
        img = np.array(Image.open(path).convert("RGB"))
        axes[i].imshow(img)
        _clean_axes(axes[i], border=False)
        axes[i].set_title(title, fontsize=10, fontweight="bold",
                          pad=6, color=COLORS["blue"])

    fig.subplots_adjust(wspace=0.05)
    _suptitle(fig, "Linear Probing Analysis: What Does the DiT Encode?", y=1.02)
    return fig


# ── Figure 11: Leakage Fix Visualization ──────────────────────────────────

def make_leakage_figure():
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("white")

    n_train_before, n_val_before = 94, 101
    leaked_before = 94
    n_train_after, n_val_after = 77, 20
    leaked_after = 0

    for ax, (n_tr, n_va, n_leak, title, period) in zip(
        [ax_before, ax_after],
        [
            (n_train_before, n_val_before, leaked_before, "Before Fix", "Random Split"),
            (n_train_after, n_val_after, leaked_after, "After Fix", "Scene-Aware Split"),
        ]
    ):
        total = n_tr + n_va
        bar_colors_train = COLORS["blue"]
        bar_colors_val = COLORS["orange"]
        bar_colors_leak = COLORS["orange"]

        bars = ax.barh(
            ["Train", "Validation", "Leaked"],
            [n_tr, n_va, n_leak],
            color=[bar_colors_train, bar_colors_val, "#c0392b" if n_leak > 0 else COLORS["green"]],
            edgecolor="white", linewidth=1.2, height=0.55
        )
        for b, v in zip(bars, [n_tr, n_va, n_leak]):
            if v > 0:
                ax.text(b.get_width() + 1, b.get_y() + b.get_height()/2,
                        str(v), ha="left", va="center", fontsize=11, fontweight="bold",
                        color=COLORS["grey"])

        ax.set_title(f"{title}\n({period})", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlim(0, max(n_train_before, n_val_before) + 15)
        ax.set_xlabel("Number of Locations", fontsize=9)
        ax.grid(axis="x", alpha=0.2, linestyle="-")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        if n_leak == 0:
            ax.text(5, 2, "ZERO leakage", fontsize=12, fontweight="bold",
                    color=COLORS["green"], va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f8f5", edgecolor=COLORS["green"], lw=1.2))
        else:
            ax.text(n_leak + 3, 2, f"{leaked_before}/{n_train_before} locations\nin both splits!",
                    fontsize=9, color="#c0392b", va="center", fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdedec", edgecolor="#c0392b", lw=1.0))

    _suptitle(fig, "Data Leakage Fix: Random → Scene-Aware Train/Val Split", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ── Figure 12: Baseline Summary Table ─────────────────────────────────────

def make_baseline_table():
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    headers = ["Method", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "FID ↓", "CFID ↓"]
    data = [
        ["Baseline (10-seed)",       "17.86 ± 0.09", "0.599 ± 0.001", "0.413 ± 0.001", "66.29 ± 0.74", "152.04 ± 1.08"],
        ["+ DPS (η = 1.0)",          "18.05",         "0.597",          "0.413",          "65.97",         "151.35"],
        ["Consistency",              "17.72",         "0.589",          "0.422",          "66.51",         "154.99"],
        ["Consistency + DPS",        "—",             "—",              "—",              "—",             "—"],
        ["", "", "", "", "", ""],
        ["Frame Ablation  (N = 1)",  "17.89",         "0.596",          "0.415",          "66.84",         "151.94"],
        ["Frame Ablation  (N = 4)",  "17.82",         "0.636",          "0.376",          "71.44",         "138.05"],
        ["Frame Ablation  (N = 16)", "16.91",         "0.651",          "0.359",          "74.85",         "131.04"],
        ["Frame Ablation  (N = 64)", "15.47",         "0.640",          "0.346",          "74.50",         "120.90"],
        ["Frame Ablation (N = 256)", "14.12",         "0.605",          "0.339",          "70.66",         "110.11"],
        ["Frame Ablation (N = 1000)","13.04",         "0.551",          "0.347",          "68.52",         "108.11"],
    ]

    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(COLORS["blue"])
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
            cell.set_height(0.08)
        elif data[r-1][0] == "":
            cell.set_facecolor("white")
            cell.set_edgecolor("white")
        elif "Baseline" in data[r-1][0]:
            cell.set_facecolor("#eaf2f8")
        elif "DPS" in data[r-1][0] and "Consist" not in data[r-1][0]:
            cell.set_facecolor("#e8f8f5")
        elif "Consistency" in data[r-1][0]:
            cell.set_facecolor("#fdf2e9")
        else:
            cell.set_facecolor("#f9f9f9" if r % 2 == 0 else "white")

    _suptitle(fig, "Quantitative Results Summary", y=0.95)
    return fig


# ── Figure 13: Contribution Boundary Table ─────────────────────────────────

def make_contribution_table():
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    headers = ["Category", "Contribution", "Lead"]
    data = [
        ["Data Capture",        "Multi-view SPAD captures with OD/RGB filters, 20k frames/view", "Mehar (hardware)"],
        ["Dataset Curation",    "GT/ISP pairing fix, hot-pixel removal, malformed data audit",     "Thesis author"],
        ["Benchmark Design",    "Scene-aware leak-free split, stratified indoor/outdoor balance",   "Thesis author"],
        ["Method",              "FLUX.1-dev + ControlNet + LoRA for 1-bit SPAD conditioning",      "Thesis author"],
        ["Physics Extension",   "Latent-space DPS with Bernoulli-motivated guidance",              "Thesis author"],
        ["Consistency Training","Cross-frame noise consistency loss (IC-Light inspired)",           "Thesis author"],
        ["Analysis",            "Linear probing of DiT blocks → bit density, depth, variance",     "Thesis author"],
        ["Evaluation",          "CFID, multi-seed UQ, frame ablation, downstream tasks",           "Thesis author"],
    ]

    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colWidths=[0.18, 0.55, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.7)

    category_colors = {
        "Data Capture": "#ebdef0",
        "Dataset Curation": "#eaf2f8",
        "Benchmark Design": "#eaf2f8",
        "Method": "#e8f8f5",
        "Physics Extension": "#e8f8f5",
        "Consistency Training": "#e8f8f5",
        "Analysis": "#fef9e7",
        "Evaluation": "#fef9e7",
    }

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(COLORS["blue"])
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
            cell.set_height(0.07)
        else:
            cat = data[r-1][0]
            cell.set_facecolor(category_colors.get(cat, "white"))
            if c == 0:
                cell.set_text_props(fontweight="bold")

    _suptitle(fig, "Contribution Boundary: Who Did What", y=0.97)
    return fig


# ── Figure 14: Probing Cross-Frame & Cross-Seed Variance Heatmaps ──────

def make_probing_variance_heatmaps():
    """Heatmaps for cross-seed variance and cross-frame variance probing R²."""
    results_path = BASE / "probing_results_allblocks" / "probes" / "probing_results.json"
    if not results_path.exists():
        print("  [SKIP] No probing results found")
        return None

    data = json.load(open(results_path))
    # Extract variance and crossframe_variance
    targets_to_plot = []
    for tname, label, cmap in [
        ("variance", "Cross-Seed Variance", "viridis"),
        ("cn_crossframe_variance", "Cross-Frame Variance (ControlNet)", "magma"),
    ]:
        if tname not in data:
            continue
        entries = data[tname]
        # Parse block names and timesteps
        blocks_set = set()
        tsteps_set = set()
        for key in entries:
            parts = key.split("_t")
            block_name = parts[0]
            tstep = int(parts[1])
            blocks_set.add(block_name)
            tsteps_set.add(tstep)
        blocks = sorted(blocks_set, key=lambda b: (0 if "joint" in b else 1, int(b.split("_")[-1])))
        tsteps = sorted(tsteps_set)

        matrix = np.full((len(blocks), len(tsteps)), np.nan)
        for bi, b in enumerate(blocks):
            for ti, t in enumerate(tsteps):
                key = f"{b}_t{t}"
                if key in entries:
                    r2 = entries[key].get("r2", np.nan)
                    matrix[bi, ti] = max(r2, 0)
        targets_to_plot.append((label, matrix, blocks, tsteps, cmap))

    if not targets_to_plot:
        return None

    n = len(targets_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 8))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    for ax, (label, matrix, blocks, tsteps, cmap) in zip(axes, targets_to_plot):
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(0.6, np.nanmax(matrix)))
        ax.set_xticks(range(len(tsteps)))
        ax.set_xticklabels(tsteps, fontsize=7)
        ax.set_xlabel("Timestep", fontsize=9)
        ax.set_yticks(range(len(blocks)))
        short_labels = [b.replace("joint_", "J").replace("single_", "S").replace("cn_joint_", "cnJ").replace("cn_single_", "cnS") for b in blocks]
        ax.set_yticklabels(short_labels, fontsize=5.5)
        ax.set_ylabel("Block", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold", pad=6, color=COLORS["blue"])
        plt.colorbar(im, ax=ax, shrink=0.6, label="R²", pad=0.02)

    _suptitle(fig, "Probing: Variance Prediction Across Blocks and Timesteps", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── Figure 15: Probing Object Recognition Bar Chart ───────────────────────

def make_probing_objects():
    """Bar chart of per-object probing accuracy (segmentation probing)."""
    results_path = BASE / "probing_results_allblocks" / "probes" / "probing_results.json"
    if not results_path.exists():
        print("  [SKIP] No probing results found")
        return None
    data = json.load(open(results_path))

    # Collect object names and their best R² across all blocks/timesteps
    obj_keys = [k for k in data.keys() if k.startswith("obj_")]
    cn_obj_keys = [k for k in data.keys() if k.startswith("cn_obj_")]

    if not obj_keys:
        print("  [SKIP] No object probing results")
        return None

    def best_r2(entries):
        best = -999
        for key, vals in entries.items():
            r2 = vals.get("r2", -999)
            if r2 > best:
                best = r2
        return max(best, 0)

    obj_names = sorted(set(k.replace("obj_", "") for k in obj_keys))
    main_r2 = [best_r2(data.get(f"obj_{name}", {})) for name in obj_names]
    cn_r2 = [best_r2(data.get(f"cn_obj_{name}", {})) for name in obj_names]

    # Sort by main model R²
    order = np.argsort(main_r2)[::-1]
    obj_names = [obj_names[i] for i in order]
    main_r2 = [main_r2[i] for i in order]
    cn_r2 = [cn_r2[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(obj_names))
    w = 0.38
    ax.bar(x - w/2, main_r2, w, color=COLORS["blue"], alpha=0.85, label="DiT (Main + LoRA)", edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, cn_r2, w, color=COLORS["orange"], alpha=0.85, label="ControlNet", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ").title() for n in obj_names], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Best R² (Object Presence)", fontsize=9)
    ax.set_xlabel("Object Category", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2, linestyle="-")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    _suptitle(fig, "Object Recognition from Activations: Per-Category R²", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── Figure 16: Improved Segmentation (multiple layout options) ─────────────

def make_seg_clean(nrows=4, layout="compact"):
    """Cleaner segmentation figure with different layouts."""
    sam3_dir = OLD_RESULTS / "sam3_montage"
    if not sam3_dir.exists():
        return None
    montage_files = sorted(sam3_dir.glob("montage_*.png"))[:nrows]
    if not montage_files:
        return None

    gt_rgb_dir = OLD_RESULTS / "ground_truth"
    out_rgb_dir = OLD_RESULTS / "output"
    input_dir = OLD_RESULTS / "input"

    if layout == "compact":
        # 3-col: GT RGB | GT Seg | Gen Seg  (no input, no gen RGB for compactness)
        fig, axes = plt.subplots(len(montage_files), 3, figsize=(9, 2.8 * len(montage_files)))
        if len(montage_files) == 1:
            axes = axes[np.newaxis, :]
        col_labels = ["Ground Truth RGB", "GT Segmentation", "Generated Segmentation"]
        hdr_colors = [COLORS["grey"], COLORS["green"], COLORS["teal"]]
        for row, mf in enumerate(montage_files):
            idx = mf.stem.split("_")[1]
            montage_img = Image.open(mf).convert("RGB")
            w_full, h_full = montage_img.size
            pw = (w_full - 4 * 20 - 30) // 5
            gap = 20; x0 = 15
            starts = [x0 + i * (pw + gap) for i in range(5)]
            crop_top = 80

            gt_rgb = _load(gt_rgb_dir / f"gt_{idx}.png")
            gt_seg = np.array(montage_img.crop((starts[3], crop_top, starts[3] + pw, h_full - 10)))
            gen_seg = np.array(montage_img.crop((starts[4], crop_top, starts[4] + pw, h_full - 10)))

            for col, img in enumerate([gt_rgb, gt_seg, gen_seg]):
                ax = axes[row, col]
                ax.imshow(img)
                _clean_axes(ax)
                if row == 0:
                    ax.set_title(col_labels[col], fontsize=9.5, fontweight="bold", pad=5, color=hdr_colors[col])
        fig.subplots_adjust(wspace=0.02, hspace=0.03)
        _suptitle(fig, "Semantic Segmentation Comparison", y=1.01)

    elif layout == "wide":
        # 5-col: SPAD | GT RGB | Gen RGB | GT Seg | Gen Seg
        fig, axes = plt.subplots(len(montage_files), 5, figsize=(14, 2.6 * len(montage_files)))
        if len(montage_files) == 1:
            axes = axes[np.newaxis, :]
        col_labels = ["SPAD Input", "GT RGB", "Generated RGB", "GT Segmentation", "Gen Segmentation"]
        hdr_colors = [COLORS["grey"], COLORS["grey"], COLORS["blue"], COLORS["green"], COLORS["teal"]]
        for row, mf in enumerate(montage_files):
            idx = mf.stem.split("_")[1]
            montage_img = Image.open(mf).convert("RGB")
            w_full, h_full = montage_img.size
            pw = (w_full - 4 * 20 - 30) // 5
            gap = 20; x0 = 15
            starts = [x0 + i * (pw + gap) for i in range(5)]
            crop_top = 80

            ctrl = _load(input_dir / f"input_{idx}.png")
            gt_rgb = _load(gt_rgb_dir / f"gt_{idx}.png")
            out_rgb = _load(out_rgb_dir / f"output_{idx}.png")
            gt_seg = np.array(montage_img.crop((starts[3], crop_top, starts[3] + pw, h_full - 10)))
            gen_seg = np.array(montage_img.crop((starts[4], crop_top, starts[4] + pw, h_full - 10)))

            for col, img in enumerate([ctrl, gt_rgb, out_rgb, gt_seg, gen_seg]):
                ax = axes[row, col]
                ax.imshow(img)
                _clean_axes(ax)
                if row == 0:
                    ax.set_title(col_labels[col], fontsize=9.5, fontweight="bold", pad=5, color=hdr_colors[col])
        fig.subplots_adjust(wspace=0.02, hspace=0.03)
        _suptitle(fig, "Semantic Segmentation: Ground Truth vs Reconstructed RGB", y=1.01)

    elif layout == "pairs":
        # Side-by-side pairs: for each row, show GT RGB|GT Seg on left, Gen RGB|Gen Seg on right
        fig, axes = plt.subplots(len(montage_files), 4, figsize=(11, 2.8 * len(montage_files)))
        if len(montage_files) == 1:
            axes = axes[np.newaxis, :]
        col_labels = ["GT RGB", "GT Segmentation", "Generated RGB", "Gen Segmentation"]
        hdr_colors = [COLORS["grey"], COLORS["green"], COLORS["blue"], COLORS["teal"]]
        for row, mf in enumerate(montage_files):
            idx = mf.stem.split("_")[1]
            montage_img = Image.open(mf).convert("RGB")
            w_full, h_full = montage_img.size
            pw = (w_full - 4 * 20 - 30) // 5
            gap = 20; x0 = 15
            starts = [x0 + i * (pw + gap) for i in range(5)]
            crop_top = 80

            gt_rgb = _load(gt_rgb_dir / f"gt_{idx}.png")
            out_rgb = _load(out_rgb_dir / f"output_{idx}.png")
            gt_seg = np.array(montage_img.crop((starts[3], crop_top, starts[3] + pw, h_full - 10)))
            gen_seg = np.array(montage_img.crop((starts[4], crop_top, starts[4] + pw, h_full - 10)))

            for col, img in enumerate([gt_rgb, gt_seg, out_rgb, gen_seg]):
                ax = axes[row, col]
                ax.imshow(img)
                _clean_axes(ax)
                if row == 0:
                    ax.set_title(col_labels[col], fontsize=9.5, fontweight="bold", pad=5, color=hdr_colors[col])
        fig.subplots_adjust(wspace=0.03, hspace=0.03)
        _suptitle(fig, "Segmentation: Ground Truth vs Reconstruction", y=1.01)

    return fig


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--hero", action="store_true")
    parser.add_argument("--pairwise", action="store_true")
    parser.add_argument("--frame-strip", action="store_true")
    parser.add_argument("--master-grid", action="store_true")
    parser.add_argument("--metric-curves", action="store_true")
    parser.add_argument("--variance", action="store_true")
    parser.add_argument("--dps-ablation", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--seg", action="store_true")
    parser.add_argument("--probing", action="store_true")
    parser.add_argument("--leakage", action="store_true")
    parser.add_argument("--baseline-table", action="store_true")
    parser.add_argument("--contribution-table", action="store_true")
    args = parser.parse_args()

    do_all = args.all or not any(vars(args).values())

    OUT.mkdir(parents=True, exist_ok=True)
    generated = []

    # ── 01_hero ──
    if do_all or args.hero:
        sub = "01_hero"
        print(f"[{sub}] Hero figures...")
        # Option A: 6-column wide
        fig = make_hero(CURATED_INDICES[:6], ncols=6)
        _save(fig, "hero_6col", generated, sub)
        # Option B: 4-column
        fig = make_hero(CURATED_INDICES[:4], ncols=4)
        _save(fig, "hero_4col", generated, sub)
        # Option C: 3-column (compact for slides)
        fig = make_hero(CURATED_INDICES[:3], ncols=3)
        _save(fig, "hero_3col", generated, sub)

    # ── 02_pairwise ──
    if do_all or args.pairwise:
        sub = "02_pairwise"
        pairs = [
            ("Baseline", BASELINE, "DPS (η=1.0)", DPS, "baseline_vs_dps"),
            ("Baseline", BASELINE, "Consistency", CONSISTENCY, "baseline_vs_consistency"),
            ("Baseline", BASELINE, "Consist.+DPS", CONSIST_DPS, "baseline_vs_consist_dps"),
            ("DPS (η=1.0)", DPS, "Consistency", CONSISTENCY, "dps_vs_consistency"),
        ]
        for a_name, a_dir, b_name, b_dir, fname in pairs:
            print(f"[{sub}] {a_name} vs {b_name}...")
            # 4-row version
            fig = make_pairwise(a_name, a_dir, b_name, b_dir, CURATED_INDICES[:4])
            _save(fig, f"{fname}_4row", generated, sub)
            # 2-row compact
            fig = make_pairwise(a_name, a_dir, b_name, b_dir, CURATED_INDICES[:2], nrows=2)
            _save(fig, f"{fname}_2row", generated, sub)

    # ── 03_frame_ablation ──
    if do_all or args.frame_strip:
        sub = "03_frame_ablation"
        print(f"[{sub}] Frame-count ablation strip...")
        fig = make_frame_strip(CURATED_INDICES[:3])
        _save(fig, "strip_3row", generated, sub)
        fig = make_frame_strip(CURATED_INDICES[:2], nrows=2)
        _save(fig, "strip_2row", generated, sub)

    # ── 04_master_grid ──
    if do_all or args.master_grid:
        sub = "04_master_grid"
        print(f"[{sub}] Master comparison grid...")
        fig = make_master_grid(CURATED_INDICES[:6])
        _save(fig, "grid_6row", generated, sub)
        fig = make_master_grid(CURATED_INDICES[:4], nrows=4)
        _save(fig, "grid_4row", generated, sub)

    # ── 05_metrics ──
    if do_all or args.metric_curves:
        sub = "05_metrics"
        print(f"[{sub}] Frame ablation metric curves...")
        fig = make_metric_curves()
        _save(fig, "frame_ablation_curves", generated, sub)

    # ── 06_dps_ablation ──
    if do_all or args.dps_ablation:
        sub = "06_dps_ablation"
        print(f"[{sub}] DPS ablation chart...")
        fig = make_dps_ablation()
        _save(fig, "dps_eta_sweep", generated, sub)

    # ── 07_variance ──
    if do_all or args.variance:
        sub = "07_variance"
        print(f"[{sub}] Variance overlay...")
        fig = make_variance_overlay(CURATED_INDICES[:6])
        _save(fig, "variance_6col", generated, sub)
        fig = make_variance_overlay(CURATED_INDICES[:4], ncols=4)
        _save(fig, "variance_4col", generated, sub)

    # ── 08_depth ──
    if do_all or args.depth:
        sub = "08_depth"
        print(f"[{sub}] Depth comparison montage...")
        fig = make_depth_montage(nrows=4)
        if fig:
            _save(fig, "depth_4row", generated, sub)
        fig = make_depth_montage(nrows=2)
        if fig:
            _save(fig, "depth_2row", generated, sub)

    # ── 09_segmentation ──
    if do_all or args.seg:
        sub = "09_segmentation"
        print(f"[{sub}] Segmentation figures (multiple layouts)...")
        # Layout A: compact (3-col: GT | GT Seg | Gen Seg)
        fig = make_seg_clean(nrows=4, layout="compact")
        if fig:
            _save(fig, "seg_compact_4row", generated, sub)
        fig = make_seg_clean(nrows=2, layout="compact")
        if fig:
            _save(fig, "seg_compact_2row", generated, sub)
        # Layout B: wide (5-col)
        fig = make_seg_clean(nrows=4, layout="wide")
        if fig:
            _save(fig, "seg_wide_4row", generated, sub)
        # Layout C: pairs (GT RGB+Seg | Gen RGB+Seg)
        fig = make_seg_clean(nrows=4, layout="pairs")
        if fig:
            _save(fig, "seg_pairs_4row", generated, sub)

    # ── 10_probing ──
    if do_all or args.probing:
        sub = "10_probing"
        print(f"[{sub}] Probing figures...")
        # Summary panel
        fig = make_probing_panel()
        if fig:
            _save(fig, "summary_panel", generated, sub)
        # Cross-frame and cross-seed variance heatmaps
        fig = make_probing_variance_heatmaps()
        if fig:
            _save(fig, "variance_heatmaps", generated, sub)
        # Object recognition bar chart
        fig = make_probing_objects()
        if fig:
            _save(fig, "object_recognition", generated, sub)
        # Copy existing probing figures into subfolder
        probing_src = BASE / "probing_analysis_output"
        if probing_src.exists():
            dst = OUT / sub
            dst.mkdir(parents=True, exist_ok=True)
            for pf in sorted(probing_src.glob("*.png")):
                shutil.copy2(pf, dst / pf.name)
            for pf in sorted(probing_src.glob("*.pdf")):
                shutil.copy2(pf, dst / pf.name)
            print(f"  -> copied 7 existing probing figures to {sub}/")

    # ── 11_leakage ──
    if do_all or args.leakage:
        sub = "11_leakage"
        print(f"[{sub}] Leakage fix visualization...")
        fig = make_leakage_figure()
        _save(fig, "leakage_fix", generated, sub)

    # ── 12_tables ──
    if do_all or args.baseline_table:
        sub = "12_tables"
        print(f"[{sub}] Baseline summary table...")
        fig = make_baseline_table()
        _save(fig, "baseline_summary", generated, sub)

    if do_all or args.contribution_table:
        sub = "12_tables"
        print(f"[{sub}] Contribution boundary table...")
        fig = make_contribution_table()
        _save(fig, "contribution_boundary", generated, sub)

    print(f"\n{'='*60}")
    print(f"Generated {len(generated)} figures in {OUT}/:")
    for f in generated:
        print(f"  {f}.pdf  /  {f}.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
