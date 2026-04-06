"""
Presentation-quality depth probe figure — 2 rows × 4 columns.
Row 1: Best indoor test example (from existing spatial files 0-99)
Row 2: Food truck outdoor scene (idx 647, extracted separately)

Designed for slides: big text, no clutter, elegant.
"""
import json, csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from PIL import Image

# ── Paths ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SPATIAL_DIR = ROOT / "probing_results" / "activations"
TARGETS_PATH = ROOT / "probing_results_allblocks" / "targets.json"
DATASET_BASE = Path("/home/jw/engsci/thesis/spad/spad_dataset")
METADATA_CSV = DATASET_BASE / "metadata_val.csv"
OUT_DIR = ROOT / "probing_analysis_output"
OUT_DIR.mkdir(exist_ok=True)

DEPTH_KEY = "single_9_t14"
VAR_KEY = "single_28_t4"
RIDGE_LAMBDA = 0.1
N_SPATIAL = 100
N_TRAIN = 80
FOOD_TRUCK_IDX = 647

# ── Typography — large for slides ───────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
})


def load_metadata():
    with open(METADATA_CSV) as f:
        return list(csv.DictReader(f))

def load_spad(meta):
    path = DATASET_BASE / meta["controlnet_image"]
    arr = np.array(Image.open(path))
    if arr.dtype == np.uint16:
        arr = (arr / 256).astype(np.uint8)
    if len(arr.shape) == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

def load_rgb(meta):
    return np.array(Image.open(DATASET_BASE / meta["image"]).convert("RGB"))


# ═══════════════════════════════════════════════════════════════════
# Train ridge probe
# ═══════════════════════════════════════════════════════════════════

def train_probe(act_key, targets_flat_all):
    """
    Two-pass streaming ridge regression.
    targets_flat_all: [N_SPATIAL, 1024] flattened spatial targets
    """
    D = 3072
    files = sorted(SPATIAL_DIR.glob("spatial_*.pt"))[:N_SPATIAL]
    train_files = files[:N_TRAIN]
    tgt = targets_flat_all[:N_TRAIN]  # [80, 1024]

    # Pass 1: normalization stats
    sum_x = np.zeros(D, dtype=np.float64)
    sum_x2 = np.zeros(D, dtype=np.float64)
    n = 0
    sum_y, ny = 0.0, 0
    for i, f in enumerate(train_files):
        x = torch.load(f, map_location="cpu", weights_only=True)[act_key].numpy().astype(np.float64)
        sum_x += x.sum(0); sum_x2 += (x**2).sum(0); n += x.shape[0]
        sum_y += tgt[i].sum(); ny += tgt[i].shape[0]
    mu = sum_x / n
    sd = np.sqrt(np.maximum(sum_x2/n - mu**2, 0)); sd = np.maximum(sd, 1e-8)
    mu_y = sum_y / ny

    # Pass 2: accumulate
    XtX = np.zeros((D, D), dtype=np.float64)
    Xty = np.zeros((D, 1), dtype=np.float64)
    for i, f in enumerate(train_files):
        x = torch.load(f, map_location="cpu", weights_only=True)[act_key].numpy().astype(np.float64)
        x = (x - mu) / sd
        y = (tgt[i] - mu_y)[:, None]
        XtX += x.T @ x; Xty += x.T @ y

    lam = RIDGE_LAMBDA * np.trace(XtX) / D
    W = np.linalg.solve(XtX + lam * np.eye(D), Xty)
    return W, mu, sd, mu_y


def predict(act_key, W, mu, sd, mu_y, spatial_tensor):
    """spatial_tensor: [1024, 3072] from .pt file"""
    x = spatial_tensor.numpy().astype(np.float64)
    x = (x - mu) / sd
    return (x @ W).squeeze() + mu_y


def r2_score(yt, yp):
    yt, yp = yt.ravel(), yp.ravel()
    ss_res = ((yt - yp)**2).sum()
    ss_tot = ((yt - yt.mean())**2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


# ═══════════════════════════════════════════════════════════════════
# Select best indoor example from test set
# ═══════════════════════════════════════════════════════════════════

def find_best_test(act_key, W, mu, sd, mu_y, targets_3d, metadata):
    """Pick best-R² test example, preferring indoor scenes."""
    files = sorted(SPATIAL_DIR.glob("spatial_*.pt"))[:N_SPATIAL]
    results = []
    for i in range(N_TRAIN, N_SPATIAL):
        d = torch.load(files[i], map_location="cpu", weights_only=True)
        pred = predict(act_key, W, mu, sd, mu_y, d[act_key]).reshape(32, 32)
        gt = targets_3d[i]
        r2 = r2_score(gt, pred)
        results.append((i, r2, pred))
    results.sort(key=lambda x: x[1], reverse=True)
    # Return the best overall
    return results[0]


# ═══════════════════════════════════════════════════════════════════
# Build the slide figure
# ═══════════════════════════════════════════════════════════════════

def make_depth_slide():
    print("Loading data...")
    metadata = load_metadata()
    with open(TARGETS_PATH) as f:
        tgt = json.load(f)
    depth_3d = np.array(tgt["spatial_depth"][:N_SPATIAL], dtype=np.float64)
    depth_flat = depth_3d.reshape(N_SPATIAL, -1)

    # Also load depth for food truck (index 647)
    depth_647 = np.array(tgt["spatial_depth"][FOOD_TRUCK_IDX], dtype=np.float64)

    print("Training depth probe...")
    W, mu, sd, mu_y = train_probe(DEPTH_KEY, depth_flat)

    # Best test example
    best_idx, best_r2, best_pred = find_best_test(DEPTH_KEY, W, mu, sd, mu_y, depth_3d, metadata)
    best_gt = depth_3d[best_idx]
    print(f"  Best test: idx={best_idx}, R²={best_r2:.3f}")

    # Food truck prediction
    ft_spatial = torch.load(OUT_DIR / f"extra_spatial_{FOOD_TRUCK_IDX:04d}.pt",
                            map_location="cpu", weights_only=True)
    ft_pred = predict(DEPTH_KEY, W, mu, sd, mu_y, ft_spatial[DEPTH_KEY]).reshape(32, 32)
    ft_r2 = r2_score(depth_647, ft_pred)
    print(f"  Food truck: idx={FOOD_TRUCK_IDX}, R²={ft_r2:.3f}")

    # ── Figure: 2 rows × 4 cols ────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 11.5))

    rows_data = [
        (best_idx, best_gt, best_pred, best_r2, "Indoor"),
        (FOOD_TRUCK_IDX, depth_647, ft_pred, ft_r2, "Outdoor"),
    ]

    col_headers = ["SPAD Input (1-bit)", "Ground Truth RGB", "GT Depth", "Predicted Depth"]

    for ri, (idx, gt_depth, pred_depth, r2, scene_label) in enumerate(rows_data):
        meta = metadata[idx]
        spad = load_spad(meta)
        rgb = load_rgb(meta)

        # Shared depth range
        d_min = min(gt_depth.min(), pred_depth.min())
        d_max = max(gt_depth.max(), pred_depth.max())

        # Col 0: SPAD
        axes[ri, 0].imshow(spad)

        # Col 1: GT RGB
        axes[ri, 1].imshow(rgb)

        # Col 2: GT Depth
        axes[ri, 2].imshow(gt_depth, cmap="inferno", vmin=d_min, vmax=d_max,
                           interpolation="bilinear")

        # Col 3: Predicted Depth
        axes[ri, 3].imshow(pred_depth, cmap="inferno", vmin=d_min, vmax=d_max,
                           interpolation="bilinear")

        # R² badge on prediction
        axes[ri, 3].text(
            0.95, 0.08, f"R² = {r2:.2f}",
            transform=axes[ri, 3].transAxes,
            fontsize=18, fontweight="bold", color="white",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6, edgecolor="none"),
        )

        # Scene label on left edge
        axes[ri, 0].text(
            -0.06, 0.5, scene_label,
            transform=axes[ri, 0].transAxes,
            fontsize=18, fontweight="bold", color="#222222",
            ha="right", va="center", rotation=90,
        )

    # Column headers
    for ci, header in enumerate(col_headers):
        axes[0, ci].set_title(header, fontsize=20, fontweight="bold", pad=14,
                              color="#222222")

    # Strip all axes
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Arrow annotation between GT depth and prediction
    for ri in range(2):
        # Draw a subtle arrow/indicator
        pass

    # Title — push suptitle higher, subtitle below it, headers below that
    fig.suptitle(
        "Depth Maps from 1-Bit Binary SPAD Data",
        fontsize=30, fontweight="bold", color="#111111", y=1.02,
    )
    fig.text(
        0.5, 0.975,
        "Linear probe on a single DiT layer  \u2014  no decoder, no depth network",
        fontsize=17, color="#555555", ha="center", style="italic",
    )

    plt.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.01,
                        wspace=0.04, hspace=0.06)

    for fmt, dpi in [("png", 200), ("pdf", None)]:
        p = OUT_DIR / f"slide_depth_from_1bit.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {p}")
    plt.close(fig)


def make_variance_slide():
    """Variance slide — 2 rows: high-uncertainty example + food truck."""
    print("\nBuilding variance slide...")
    metadata = load_metadata()
    with open(TARGETS_PATH) as f:
        tgt = json.load(f)
    var_3d = np.array(tgt["spatial_variance"][:N_SPATIAL], dtype=np.float64)
    var_flat = var_3d.reshape(N_SPATIAL, -1)
    var_647 = np.array(tgt["spatial_variance"][FOOD_TRUCK_IDX], dtype=np.float64)

    RECON_DIR = ROOT / "validation_outputs_scene_aware" / "seed_42" / "output"

    print("Training variance probe...")
    W, mu, sd, mu_y = train_probe(VAR_KEY, var_flat)

    # Pick idx=87: high uncertainty (var_mean=0.147) + good R² (0.619)
    PICK_IDX = 87
    d = torch.load(sorted(SPATIAL_DIR.glob("spatial_*.pt"))[PICK_IDX],
                    map_location="cpu", weights_only=True)
    x = d[VAR_KEY].numpy().astype(np.float64)
    x = (x - mu) / sd
    best_pred = ((x @ W).squeeze() + mu_y).reshape(32, 32)
    best_gt = var_3d[PICK_IDX]
    best_r2 = r2_score(best_gt, best_pred)
    best_idx = PICK_IDX
    print(f"  High-var pick: idx={best_idx}, R²={best_r2:.3f}, var_mean={best_gt.mean():.4f}")

    # Food truck
    ft_spatial = torch.load(OUT_DIR / f"extra_spatial_{FOOD_TRUCK_IDX:04d}.pt",
                            map_location="cpu", weights_only=True)
    ft_pred = predict(VAR_KEY, W, mu, sd, mu_y, ft_spatial[VAR_KEY]).reshape(32, 32)
    ft_r2 = r2_score(var_647, ft_pred)
    print(f"  Food truck: idx={FOOD_TRUCK_IDX}, R²={ft_r2:.3f}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 11.5))
    col_headers = ["SPAD Input (1-bit)", "Reconstruction", "GT Variance", "Predicted Variance"]

    rows_data = [
        (best_idx, best_gt, best_pred, best_r2, "Indoor\n(complex)"),
        (FOOD_TRUCK_IDX, var_647, ft_pred, ft_r2, "Outdoor"),
    ]

    for ri, (idx, gt_var, pred_var, r2, scene_label) in enumerate(rows_data):
        meta = metadata[idx]
        spad = load_spad(meta)
        recon = np.array(Image.open(RECON_DIR / f"output_{idx:04d}.png").convert("RGB"))
        pred_clipped = np.clip(pred_var, 0, None)

        v_min = 0
        v_max = max(gt_var.max(), pred_clipped.max())

        axes[ri, 0].imshow(spad)
        axes[ri, 1].imshow(recon)
        axes[ri, 2].imshow(gt_var, cmap="hot", vmin=v_min, vmax=v_max, interpolation="bilinear")
        axes[ri, 3].imshow(pred_clipped, cmap="hot", vmin=v_min, vmax=v_max, interpolation="bilinear")

        axes[ri, 3].text(
            0.95, 0.08, f"R² = {r2:.2f}",
            transform=axes[ri, 3].transAxes,
            fontsize=18, fontweight="bold", color="white",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6, edgecolor="none"),
        )
        axes[ri, 0].text(
            -0.06, 0.5, scene_label,
            transform=axes[ri, 0].transAxes,
            fontsize=18, fontweight="bold", color="#222222",
            ha="right", va="center", rotation=90,
        )

    for ci, header in enumerate(col_headers):
        axes[0, ci].set_title(header, fontsize=20, fontweight="bold", pad=14, color="#222222")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.suptitle(
        "The Model Knows Where It's Uncertain",
        fontsize=30, fontweight="bold", color="#111111", y=1.02,
    )
    fig.text(
        0.5, 0.975,
        "Linear probe predicts per-pixel reconstruction variance from a single layer",
        fontsize=17, color="#555555", ha="center", style="italic",
    )
    plt.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.01,
                        wspace=0.04, hspace=0.06)

    for fmt, dpi in [("png", 200), ("pdf", None)]:
        p = OUT_DIR / f"slide_variance_awareness.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    make_depth_slide()
    make_variance_slide()
    print("\nDone — all slides in", OUT_DIR)
