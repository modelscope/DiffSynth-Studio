"""
Publication-quality 4-panel linear probing heatmap.
Panels: Bit Density, Depth, Seed Variance, Cross-Frame Variance
Rows: J0–J18 (joint) then S0–S37 (single) — 57 blocks
Cols: timesteps 0, 4, 9, 14, 19, 24, 27
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.patheffects as patheffects
from pathlib import Path

# ── Typography ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
})

# ── Load data ───────────────────────────────────────────────────────
DATA_PATH = Path("probing_results_allblocks/probes/probing_results.json")
OUT_DIR = Path("probing_analysis_output")
OUT_DIR.mkdir(exist_ok=True)

with open(DATA_PATH) as f:
    data = json.load(f)

TARGETS = [
    ("bit_density",          "Bit Density"),
    ("depth",                "Depth"),
    ("variance",             "Seed Variance"),
    ("crossframe_variance",  "Cross-Frame Variance"),
]
TIMESTEPS = [0, 4, 9, 14, 19, 24, 27]
JOINT_BLOCKS = list(range(19))    # 0..18
SINGLE_BLOCKS = list(range(38))   # 0..37

# Build ordered block list: joint first, then single
block_keys = []          # e.g. "joint_0", "single_0"
block_labels_full = []   # e.g. "J0", "S0"
for i in JOINT_BLOCKS:
    block_keys.append(f"joint_{i}")
    block_labels_full.append(f"J{i}")
for i in SINGLE_BLOCKS:
    block_keys.append(f"single_{i}")
    block_labels_full.append(f"S{i}")

N_BLOCKS = len(block_keys)   # 57
N_TIMESTEPS = len(TIMESTEPS) # 7

# ── Build matrices ──────────────────────────────────────────────────
matrices = {}
for target_key, _ in TARGETS:
    mat = np.full((N_BLOCKS, N_TIMESTEPS), np.nan)
    tgt_data = data[target_key]
    for bi, bk in enumerate(block_keys):
        for ti, ts in enumerate(TIMESTEPS):
            key = f"{bk}_t{ts}"
            if key in tgt_data:
                mat[bi, ti] = tgt_data[key]["r2"]
    matrices[target_key] = mat

# ── Y-axis tick selection ───────────────────────────────────────────
# Label every 3rd joint, every 5th single, plus S37 (last)
y_tick_positions = []
y_tick_labels = []
for i in JOINT_BLOCKS:
    if i % 3 == 0:
        y_tick_positions.append(i)
        y_tick_labels.append(f"J{i}")
for i in SINGLE_BLOCKS:
    idx = len(JOINT_BLOCKS) + i
    if i % 5 == 0 or i == 37:
        y_tick_positions.append(idx)
        y_tick_labels.append(f"S{i}")

# ── Joint/Single boundary ──────────────────────────────────────────
boundary_y = len(JOINT_BLOCKS) - 0.5  # between J18 and S0

# ── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 10))

# Outer gridspec: 4 columns with generous spacing
# Each column is split into [heatmap | thin colorbar]
outer_gs = gridspec.GridSpec(
    1, 4, figure=fig,
    left=0.06, right=0.98, top=0.88, bottom=0.07,
    wspace=0.25,
)

axes_heat = []
axes_cbar = []
for col in range(4):
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[col],
        width_ratios=[1, 0.045], wspace=0.04,
    )
    ax_h = fig.add_subplot(inner[0])
    ax_c = fig.add_subplot(inner[1])
    axes_heat.append(ax_h)
    axes_cbar.append(ax_c)

# ── Plot each panel ─────────────────────────────────────────────────
for pi, (target_key, title) in enumerate(TARGETS):
    ax = axes_heat[pi]
    cax = axes_cbar[pi]
    mat = matrices[target_key].copy()

    # Clip negatives to 0
    mat_clipped = np.clip(mat, 0, None)

    # Per-panel color range
    vmin = 0.0
    vmax = np.nanmax(mat_clipped)
    # For bit_density, tighten range so structure is visible
    if target_key == "bit_density":
        valid = mat_clipped[~np.isnan(mat_clipped)]
        vmin = max(0, np.percentile(valid, 2) - 0.02)

    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        mat_clipped,
        aspect="auto",
        cmap="magma",
        norm=norm,
        interpolation="nearest",
    )

    # Colorbar
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$R^2$", fontsize=11, fontstyle="italic")
    cb.ax.tick_params(labelsize=9)

    # Joint/Single boundary line
    ax.axhline(y=boundary_y, color="white", linewidth=2.5, linestyle="-")

    # ── Peak annotation ─────────────────────────────────────────────
    peak_val = np.nanmax(mat_clipped)
    peak_idx = np.unravel_index(np.nanargmax(mat_clipped), mat_clipped.shape)
    peak_row, peak_col = peak_idx

    # Star marker with dark edge for contrast on both light and dark cells
    ax.plot(
        peak_col, peak_row,
        marker="*", color="white", markersize=14,
        markeredgecolor="black", markeredgewidth=0.6,
        zorder=10,
    )

    # Place text below star (or above if near bottom edge)
    text_y = peak_row + 3.0 if peak_row < N_BLOCKS - 6 else peak_row - 3.0
    # Use text with a subtle shadow for legibility on any background
    ax.text(
        peak_col, text_y,
        f"{peak_val:.3f}",
        fontsize=10, fontweight="bold", color="white",
        ha="center", va="center",
        path_effects=[
            patheffects.withStroke(linewidth=2.5, foreground="black"),
        ],
        zorder=10,
    )

    # ── Axes ────────────────────────────────────────────────────────
    ax.set_xticks(range(N_TIMESTEPS))
    ax.set_xticklabels([str(t) for t in TIMESTEPS], fontsize=10)
    ax.set_xlabel("Timestep", fontsize=12)

    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, fontsize=10)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)

    # Only leftmost panel shows y-tick labels
    if pi != 0:
        ax.set_yticklabels([])

    ax.tick_params(axis="both", which="both", length=0)  # no tick marks

# ── Joint / Single margin annotations ───────────────────────────────
ax0 = axes_heat[0]
joint_center_frac = ((len(JOINT_BLOCKS) - 1) / 2.0) / (N_BLOCKS - 1)
single_center_frac = (len(JOINT_BLOCKS) + (len(SINGLE_BLOCKS) - 1) / 2.0) / (N_BLOCKS - 1)

for label, y_frac in [("Joint", joint_center_frac), ("Single", single_center_frac)]:
    ax0.text(
        -0.15, y_frac, label,
        transform=ax0.transAxes,
        fontsize=13, fontweight="bold", color="#333333",
        ha="center", va="center", rotation=90,
    )

# ── Suptitle ────────────────────────────────────────────────────────
fig.suptitle(
    "Linear Probe $R^2$ — Main Model (LoRA)",
    fontsize=20, fontweight="bold", y=0.96,
)

# ── Save ────────────────────────────────────────────────────────────
png_path = OUT_DIR / "fig1_main_heatmap.png"
pdf_path = OUT_DIR / "fig1_main_heatmap.pdf"
fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# Print peak info
for target_key, title in TARGETS:
    mat = np.clip(matrices[target_key], 0, None)
    peak_val = np.nanmax(mat)
    peak_idx = np.unravel_index(np.nanargmax(mat), mat.shape)
    block_name = block_labels_full[peak_idx[0]]
    ts = TIMESTEPS[peak_idx[1]]
    print(f"  {title:25s}  peak R²={peak_val:.4f}  at {block_name} t={ts}")
