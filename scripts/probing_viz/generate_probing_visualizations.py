"""
Generate publication-quality probing visualizations for presentation.

Figure 1: Spatial depth prediction — "Depth from 1-bit data"
Figure 2: Spatial variance prediction — "The model knows where it's uncertain"
Figure 3: Probing scorecard table

Trains ridge probes from spatial activations, generates predictions,
and produces visual comparison grids.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from PIL import Image

# ── Paths ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SPATIAL_DIR = ROOT / "probing_results" / "activations"
TARGETS_PATH = ROOT / "probing_results_allblocks" / "targets.json"
DATASET_BASE = Path("/home/jw/engsci/thesis/spad/spad_dataset")
METADATA_CSV = DATASET_BASE / "metadata_val.csv"
RECON_DIR = ROOT / "validation_outputs_scene_aware" / "seed_42" / "output"
OUT_DIR = ROOT / "probing_analysis_output"
OUT_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────
DEPTH_KEY = "single_9_t14"     # Best spatial depth R²=0.685
VAR_KEY = "single_28_t4"       # Best spatial variance R²=0.506
RIDGE_LAMBDA = 0.1
N_SPATIAL = 100                # Number of spatial activation files
N_TRAIN = 80                   # 80/20 split
N_TEST = N_SPATIAL - N_TRAIN

# ── Typography ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
})


# ═══════════════════════════════════════════════════════════════════
# PART 0: Load metadata and targets
# ═══════════════════════════════════════════════════════════════════

def load_metadata():
    """Load CSV metadata for image paths."""
    import csv
    rows = []
    with open(METADATA_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def load_targets():
    """Load spatial probe targets (32x32 maps)."""
    with open(TARGETS_PATH) as f:
        targets = json.load(f)
    spatial_depth = np.array(targets["spatial_depth"][:N_SPATIAL], dtype=np.float64)   # [100, 32, 32]
    spatial_var = np.array(targets["spatial_variance"][:N_SPATIAL], dtype=np.float64)  # [100, 32, 32]
    return spatial_depth, spatial_var


# ═══════════════════════════════════════════════════════════════════
# PART 1: Train ridge probes via streaming accumulation
# ═══════════════════════════════════════════════════════════════════

def train_spatial_probe(act_key, targets_3d):
    """
    Train a spatial ridge probe via two-pass streaming.

    Args:
        act_key: key in .pt files, e.g. 'single_9_t14'
        targets_3d: [N_SPATIAL, 32, 32] target maps

    Returns:
        W: [3072, 1] weight vector
        mu_feat: [3072] feature means
        sd_feat: [3072] feature stds
        mu_y: scalar target mean
    """
    D = 3072
    spatial_files = sorted(SPATIAL_DIR.glob("spatial_*.pt"))[:N_SPATIAL]
    assert len(spatial_files) == N_SPATIAL, f"Expected {N_SPATIAL} files, got {len(spatial_files)}"

    train_files = spatial_files[:N_TRAIN]
    targets_flat = targets_3d[:N_TRAIN].reshape(N_TRAIN, -1)  # [80, 1024]

    # ── Pass 1: Feature mean/std + target mean ──────────────────────
    print(f"  Pass 1: computing normalization stats for {act_key}...")
    sum_x = np.zeros(D, dtype=np.float64)
    sum_x2 = np.zeros(D, dtype=np.float64)
    n_tokens = 0
    sum_y = 0.0
    n_y = 0

    for i, fpath in enumerate(train_files):
        d = torch.load(fpath, map_location="cpu", weights_only=True)
        x = d[act_key].numpy().astype(np.float64)  # [1024, 3072]
        sum_x += x.sum(axis=0)
        sum_x2 += (x ** 2).sum(axis=0)
        n_tokens += x.shape[0]

        y = targets_flat[i]  # [1024]
        sum_y += y.sum()
        n_y += y.shape[0]

    mu_feat = sum_x / n_tokens
    sd_feat = np.sqrt(sum_x2 / n_tokens - mu_feat ** 2)
    sd_feat = np.maximum(sd_feat, 1e-8)  # avoid div-by-zero
    mu_y = sum_y / n_y

    # ── Pass 2: Accumulate XᵀX and Xᵀy ─────────────────────────────
    print(f"  Pass 2: accumulating XᵀX, Xᵀy...")
    XtX = np.zeros((D, D), dtype=np.float64)
    Xty = np.zeros((D, 1), dtype=np.float64)

    for i, fpath in enumerate(train_files):
        d = torch.load(fpath, map_location="cpu", weights_only=True)
        x = d[act_key].numpy().astype(np.float64)  # [1024, 3072]
        x = (x - mu_feat) / sd_feat  # normalize

        y = targets_flat[i] - mu_y  # [1024], centered
        y = y[:, None]              # [1024, 1]

        XtX += x.T @ x
        Xty += x.T @ y

    # ── Solve ridge regression ──────────────────────────────────────
    print(f"  Solving ridge (lambda={RIDGE_LAMBDA})...")
    trace_XtX = np.trace(XtX)
    lam_scaled = RIDGE_LAMBDA * trace_XtX / D
    W = np.linalg.solve(XtX + lam_scaled * np.eye(D), Xty)  # [3072, 1]

    print(f"  Done. W shape: {W.shape}, |W|_2 = {np.linalg.norm(W):.4f}")
    return W, mu_feat, sd_feat, mu_y


def predict_spatial(act_key, W, mu_feat, sd_feat, mu_y, idx):
    """Predict 32x32 spatial map for a single test image."""
    fpath = sorted(SPATIAL_DIR.glob("spatial_*.pt"))[idx]
    d = torch.load(fpath, map_location="cpu", weights_only=True)
    x = d[act_key].numpy().astype(np.float64)  # [1024, 3072]
    x = (x - mu_feat) / sd_feat
    y_pred = (x @ W).squeeze() + mu_y  # [1024]
    return y_pred.reshape(32, 32)


def per_sample_r2(y_true_32, y_pred_32):
    """Compute R² for a single 32x32 map."""
    yt = y_true_32.ravel()
    yp = y_pred_32.ravel()
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ═══════════════════════════════════════════════════════════════════
# PART 2: Load images
# ═══════════════════════════════════════════════════════════════════

def load_spad_image(meta_row):
    """Load SPAD binary input, convert to displayable grayscale."""
    path = DATASET_BASE / meta_row["controlnet_image"]
    img = Image.open(path)
    arr = np.array(img)
    if arr.dtype == np.uint16:
        arr = (arr / 256).astype(np.uint8)
    if len(arr.shape) == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr

def load_rgb_image(meta_row):
    """Load ground truth RGB."""
    path = DATASET_BASE / meta_row["image"]
    return np.array(Image.open(path).convert("RGB"))

def load_reconstruction(idx):
    """Load reconstruction output from seed_42."""
    path = RECON_DIR / f"output_{idx:04d}.png"
    return np.array(Image.open(path).convert("RGB"))


# ═══════════════════════════════════════════════════════════════════
# PART 3: Select best examples
# ═══════════════════════════════════════════════════════════════════

def select_examples(all_r2, targets_3d):
    """
    Pick 4 examples from test set:
    - 1st best R²
    - 2nd best R²
    - Median R²
    - Most sparse SPAD (lowest bit density in targets → darkest input)
    """
    test_indices = list(range(N_TRAIN, N_SPATIAL))  # 80..99
    r2_list = [(test_indices[i], all_r2[i]) for i in range(N_TEST)]
    r2_sorted = sorted(r2_list, key=lambda x: x[1], reverse=True)

    best1_idx, best1_r2 = r2_sorted[0]
    best2_idx, best2_r2 = r2_sorted[1]
    median_pos = len(r2_sorted) // 2
    median_idx, median_r2 = r2_sorted[median_pos]

    # Find sparsest SPAD: lowest mean target value in the test set
    # Use bit density from targets if available, otherwise use depth range as proxy
    target_means = []
    for i, (global_idx, _) in enumerate(r2_list):
        tmap = targets_3d[global_idx]
        target_means.append((global_idx, tmap.mean(), all_r2[i]))

    # Sort by target map mean (ascending = lowest density / darkest)
    target_means_sorted = sorted(target_means, key=lambda x: x[1])
    # Pick the sparsest that isn't already selected
    selected = {best1_idx, best2_idx, median_idx}
    sparse_idx, sparse_mean, sparse_r2 = None, None, None
    for gidx, tmean, r2 in target_means_sorted:
        if gidx not in selected:
            sparse_idx, sparse_mean, sparse_r2 = gidx, tmean, r2
            break

    picks = [
        (best1_idx, f"Best (R²={best1_r2:.3f})"),
        (best2_idx, f"2nd Best (R²={best2_r2:.3f})"),
        (median_idx, f"Median (R²={median_r2:.3f})"),
        (sparse_idx, f"Hard Case (R²={sparse_r2:.3f})"),
    ]
    print(f"  Selected examples: {[p[0] for p in picks]}")
    return picks


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Spatial Depth Predictions
# ═══════════════════════════════════════════════════════════════════

def make_depth_figure(depth_W, depth_mu_feat, depth_sd_feat, depth_mu_y,
                      spatial_depth, metadata):
    """4-row × 4-col: SPAD | GT RGB | GT Depth | Predicted Depth"""
    print("\n=== Figure 1: Depth from 1-bit data ===")

    # Compute predictions and R² for all test samples
    test_r2 = []
    test_preds = []
    for i in range(N_TEST):
        idx = N_TRAIN + i
        pred = predict_spatial(DEPTH_KEY, depth_W, depth_mu_feat, depth_sd_feat, depth_mu_y, idx)
        gt = spatial_depth[idx]
        r2 = per_sample_r2(gt, pred)
        test_r2.append(r2)
        test_preds.append(pred)

    overall_r2 = np.mean(test_r2)
    print(f"  Test R² mean: {overall_r2:.4f}, min: {min(test_r2):.4f}, max: {max(test_r2):.4f}")

    picks = select_examples(test_r2, spatial_depth)

    # ── Build figure ────────────────────────────────────────────────
    n_rows = 4
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))

    col_headers = ["SPAD Input", "Ground Truth", "GT Depth",
                   f"Probe Prediction"]

    for row_i, (global_idx, label) in enumerate(picks):
        test_i = global_idx - N_TRAIN
        pred = test_preds[test_i]
        gt_depth = spatial_depth[global_idx]
        r2 = test_r2[test_i]
        meta = metadata[global_idx]

        spad = load_spad_image(meta)
        rgb = load_rgb_image(meta)

        # Shared depth color range for this row
        d_min = min(gt_depth.min(), pred.min())
        d_max = max(gt_depth.max(), pred.max())

        # Col 0: SPAD input
        axes[row_i, 0].imshow(spad)

        # Col 1: GT RGB
        axes[row_i, 1].imshow(rgb)

        # Col 2: GT Depth
        axes[row_i, 2].imshow(gt_depth, cmap="inferno", vmin=d_min, vmax=d_max,
                               interpolation="nearest")

        # Col 3: Predicted Depth
        axes[row_i, 3].imshow(pred, cmap="inferno", vmin=d_min, vmax=d_max,
                               interpolation="nearest")

        # R² annotation on prediction panel
        axes[row_i, 3].text(
            0.97, 0.05, f"R²={r2:.3f}",
            transform=axes[row_i, 3].transAxes,
            fontsize=11, fontweight="bold", color="white",
            ha="right", va="bottom",
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="black")],
        )

        # Row label on left
        axes[row_i, 0].text(
            -0.05, 0.5, label,
            transform=axes[row_i, 0].transAxes,
            fontsize=10, fontweight="bold", color="#333333",
            ha="right", va="center", rotation=90,
        )

    # Column headers
    for ci, header in enumerate(col_headers):
        axes[0, ci].set_title(header, fontsize=14, fontweight="bold", pad=8)

    # Remove all axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Emergent Depth Encoding from 1-Bit Binary Data",
        fontsize=20, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.945,
        "Linear probe on single layer activations (block S9, timestep 14/28)",
        fontsize=13, color="#555555", ha="center",
    )

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.02,
                        wspace=0.04, hspace=0.06)

    for fmt, dpi in [("png", 200), ("pdf", None)]:
        path = OUT_DIR / f"spatial_depth_predictions.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Spatial Variance Predictions
# ═══════════════════════════════════════════════════════════════════

def make_variance_figure(var_W, var_mu_feat, var_sd_feat, var_mu_y,
                         spatial_var, metadata):
    """4-row × 4-col: SPAD | Reconstruction | GT Variance | Predicted Variance"""
    print("\n=== Figure 2: Uncertainty Self-Awareness ===")

    test_r2 = []
    test_preds = []
    for i in range(N_TEST):
        idx = N_TRAIN + i
        pred = predict_spatial(VAR_KEY, var_W, var_mu_feat, var_sd_feat, var_mu_y, idx)
        gt = spatial_var[idx]
        r2 = per_sample_r2(gt, pred)
        test_r2.append(r2)
        test_preds.append(pred)

    overall_r2 = np.mean(test_r2)
    print(f"  Test R² mean: {overall_r2:.4f}, min: {min(test_r2):.4f}, max: {max(test_r2):.4f}")

    picks = select_examples(test_r2, spatial_var)

    n_rows = 4
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))

    col_headers = ["SPAD Input", "Reconstruction", "GT Variance",
                   f"Probe Prediction"]

    for row_i, (global_idx, label) in enumerate(picks):
        test_i = global_idx - N_TRAIN
        pred = test_preds[test_i]
        gt_var = spatial_var[global_idx]
        r2 = test_r2[test_i]
        meta = metadata[global_idx]

        spad = load_spad_image(meta)
        recon = load_reconstruction(global_idx)

        # Clip predicted variance to non-negative
        pred_clipped = np.clip(pred, 0, None)

        # Shared variance color range
        v_min = 0
        v_max = max(gt_var.max(), pred_clipped.max())

        # Col 0: SPAD
        axes[row_i, 0].imshow(spad)

        # Col 1: Reconstruction
        axes[row_i, 1].imshow(recon)

        # Col 2: GT Variance
        axes[row_i, 2].imshow(gt_var, cmap="hot", vmin=v_min, vmax=v_max,
                               interpolation="nearest")

        # Col 3: Predicted Variance
        axes[row_i, 3].imshow(pred_clipped, cmap="hot", vmin=v_min, vmax=v_max,
                               interpolation="nearest")

        # R² annotation
        axes[row_i, 3].text(
            0.97, 0.05, f"R²={r2:.3f}",
            transform=axes[row_i, 3].transAxes,
            fontsize=11, fontweight="bold", color="white",
            ha="right", va="bottom",
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="black")],
        )

        axes[row_i, 0].text(
            -0.05, 0.5, label,
            transform=axes[row_i, 0].transAxes,
            fontsize=10, fontweight="bold", color="#333333",
            ha="right", va="center", rotation=90,
        )

    for ci, header in enumerate(col_headers):
        axes[0, ci].set_title(header, fontsize=14, fontweight="bold", pad=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Uncertainty Self-Awareness: The Model Knows Where It's Guessing",
        fontsize=20, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.945,
        "Linear probe on single layer activations (block S28, timestep 4/28)",
        fontsize=13, color="#555555", ha="center",
    )

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.02,
                        wspace=0.04, hspace=0.06)

    for fmt, dpi in [("png", 200), ("pdf", None)]:
        path = OUT_DIR / f"spatial_variance_predictions.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Probing Scorecard
# ═══════════════════════════════════════════════════════════════════

def make_scorecard():
    """Publication-quality table comparing LoRA vs no-LoRA vs no-ControlNet."""
    print("\n=== Figure 3: Probing Scorecard ===")

    # Data rows: (property, with_lora, without_lora, no_controlnet, helps, delta)
    rows = [
        ("Bit Density",       "0.998",  "0.999",  "\u22120.059",  False,  "~0"),
        ("Depth (spatial)",   "0.685",  "0.453",  "\u22120.040",  True,   "+0.232"),
        ("Seed Variance",     "0.506",  "0.434",  "\u22120.067",  True,   "+0.072"),
        ("Cross-Frame Var.",  "0.292",  "0.222",  "\u22120.161",  True,   "+0.070"),
        ("Objects (table)",   "98.2%",  "91.2%",  "\u2014",       True,   "+7.0%"),
    ]

    headers = ["Property", "With LoRA", "Without LoRA", "No ControlNet", "LoRA helps?"]
    col_widths = [0.22, 0.16, 0.18, 0.20, 0.24]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_rows = len(rows) + 1  # +1 for header
    row_height = 0.12
    y_start = 0.88

    # Color palette
    c_green = "#d4edda"
    c_yellow = "#fff3cd"
    c_red = "#f8d7da"
    c_header = "#343a40"
    c_header_text = "#ffffff"
    c_row_alt = "#f8f9fa"

    def val_color(text):
        """Return cell background color based on value."""
        text_clean = text.replace("\u2212", "-").replace("%", "").replace("\u2014", "")
        try:
            v = float(text_clean)
            if v > 0.5:
                return c_green
            elif v > 0.1:
                return c_yellow
            elif v < 0:
                return c_red
            else:
                return c_yellow
        except ValueError:
            return None

    # ── Draw header ─────────────────────────────────────────────────
    y = y_start
    x_pos = 0.0
    for ci, (header, w) in enumerate(zip(headers, col_widths)):
        rect = plt.Rectangle((x_pos, y - row_height * 0.5), w - 0.005, row_height,
                              facecolor=c_header, edgecolor="white", linewidth=1.5,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x_pos + w / 2, y, header,
                transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=c_header_text,
                ha="center", va="center")
        x_pos += w

    # ── Draw data rows ──────────────────────────────────────────────
    for ri, (prop, wl, wol, ncn, helps, delta) in enumerate(rows):
        y = y_start - (ri + 1) * row_height
        cells = [prop, wl, wol, ncn, ""]
        x_pos = 0.0

        for ci, (cell, w) in enumerate(zip(cells, col_widths)):
            if ci == 0:
                bg = c_row_alt if ri % 2 == 0 else "white"
            elif ci == 4:
                bg = c_row_alt if ri % 2 == 0 else "white"
            else:
                vc = val_color(cell)
                bg = vc if vc else (c_row_alt if ri % 2 == 0 else "white")

            rect = plt.Rectangle((x_pos, y - row_height * 0.5), w - 0.005, row_height,
                                  facecolor=bg, edgecolor="#dee2e6", linewidth=0.8,
                                  transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)

            # Cell text
            if ci == 0:
                ax.text(x_pos + 0.01, y, cell,
                        transform=ax.transAxes,
                        fontsize=11, fontweight="bold", color="#212529",
                        ha="left", va="center")
            elif ci < 4:
                ax.text(x_pos + w / 2, y, cell,
                        transform=ax.transAxes,
                        fontsize=11, color="#212529",
                        ha="center", va="center",
                        fontfamily="monospace")
            x_pos += w

        # "LoRA helps?" column — special formatting
        x_last = sum(col_widths[:4])
        w_last = col_widths[4]
        if helps:
            check_text = f"\u2713 {delta}"
            check_color = "#28a745"
        else:
            check_text = f"No ({delta})"
            check_color = "#6c757d"

        ax.text(x_last + w_last / 2, y, check_text,
                transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=check_color,
                ha="center", va="center")

    # ── Title ───────────────────────────────────────────────────────
    ax.text(0.5, 1.08, "Linear Probing Scorecard",
            transform=ax.transAxes,
            fontsize=18, fontweight="bold", color="#212529", ha="center")
    ax.text(0.5, 1.01,
            "Best R² (or balanced accuracy) across all blocks and timesteps",
            transform=ax.transAxes,
            fontsize=11, color="#6c757d", ha="center")

    for fmt, dpi in [("png", 200), ("pdf", None)]:
        path = OUT_DIR / f"probing_scorecard.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("Loading metadata and targets...")
    metadata = load_metadata()
    spatial_depth, spatial_var = load_targets()
    print(f"  Spatial depth: {spatial_depth.shape}")
    print(f"  Spatial variance: {spatial_var.shape}")

    # ── Train depth probe ───────────────────────────────────────────
    print(f"\nTraining depth probe ({DEPTH_KEY})...")
    depth_W, depth_mu, depth_sd, depth_muy = train_spatial_probe(
        DEPTH_KEY, spatial_depth
    )

    # ── Train variance probe ────────────────────────────────────────
    print(f"\nTraining variance probe ({VAR_KEY})...")
    var_W, var_mu, var_sd, var_muy = train_spatial_probe(
        VAR_KEY, spatial_var
    )

    # ── Generate figures ────────────────────────────────────────────
    make_depth_figure(depth_W, depth_mu, depth_sd, depth_muy,
                      spatial_depth, metadata)

    make_variance_figure(var_W, var_mu, var_sd, var_muy,
                         spatial_var, metadata)

    make_scorecard()

    print("\n" + "=" * 60)
    print("All figures saved to:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
