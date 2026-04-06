#!/usr/bin/env python3
"""
Comprehensive probing analysis for SPAD linear probing experiments.
Generates publication-quality figures comparing Main (LoRA), Control (no LoRA),
and optionally No-ControlNet conditions.

Usage:
    /home/jw/miniconda3/envs/diffsynth/bin/python probing_analysis.py
"""

import matplotlib
matplotlib.use("Agg")

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "probing_analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULT_PATHS = {
    "Main (LoRA)": BASE_DIR / "probing_results_allblocks" / "probes" / "probing_results.json",
    "Control (no LoRA)": BASE_DIR / "probing_results_control" / "probes" / "probing_results.json",
    "No ControlNet": BASE_DIR / "probing_results_no_cn" / "probes" / "probing_results.json",
}

# DiT architecture
N_JOINT = 19   # joint_0 .. joint_18
N_SINGLE = 38  # single_0 .. single_37
N_DIT = N_JOINT + N_SINGLE  # 57

# ControlNet architecture
N_CN_JOINT = 5   # cn_joint_0 .. cn_joint_4
N_CN_SINGLE = 10 # cn_single_0 .. cn_single_9
N_CN = N_CN_JOINT + N_CN_SINGLE  # 15

# Timesteps (in order)
TIMESTEPS = ["t0", "t4", "t9", "t14", "t19", "t24", "t27"]
TIMESTEP_LABELS = ["0", "4", "9", "14", "19", "24", "27"]
N_TIMESTEPS = len(TIMESTEPS)

# Block ordering for DiT
DIT_BLOCKS = [f"joint_{i}" for i in range(N_JOINT)] + \
             [f"single_{i}" for i in range(N_SINGLE)]

# Block ordering for CN
CN_BLOCKS = [f"cn_joint_{i}" for i in range(N_CN_JOINT)] + \
            [f"cn_single_{i}" for i in range(N_CN_SINGLE)]

# Continuous targets and their display names
CONTINUOUS_TARGETS = ["bit_density", "depth", "variance"]
TARGET_DISPLAY = {
    "bit_density": "Bit Density",
    "depth": "Depth",
    "variance": "Variance",
}

# Spatial counterparts
SPATIAL_TARGETS = ["spatial_bit_density", "spatial_depth", "spatial_variance"]

# CN counterparts
CN_TARGETS = ["cn_bit_density", "cn_depth", "cn_variance"]

# Spatial blocks (subset used for spatial probing)
SPATIAL_JOINT_INDICES = [0, 4, 9, 14, 18]
SPATIAL_SINGLE_INDICES = [0, 9, 19, 28, 37]
SPATIAL_DIT_BLOCKS = [f"joint_{i}" for i in SPATIAL_JOINT_INDICES] + \
                     [f"single_{i}" for i in SPATIAL_SINGLE_INDICES]

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_results():
    """Load all available result files."""
    results = {}
    for label, path in RESULT_PATHS.items():
        if path.exists():
            with open(path) as f:
                results[label] = json.load(f)
            print(f"  Loaded: {label} ({path})")
        else:
            print(f"  Skipped (not found): {label} ({path})")
    return results


def extract_heatmap(data, target, blocks, timesteps):
    """
    Extract a 2-D array (n_blocks x n_timesteps) of R2 values.
    Returns np.nan for missing entries.
    """
    n_blocks = len(blocks)
    n_ts = len(timesteps)
    mat = np.full((n_blocks, n_ts), np.nan)
    if target not in data:
        return mat
    target_data = data[target]
    for bi, block in enumerate(blocks):
        for ti, ts in enumerate(timesteps):
            key = f"{block}_{ts}"
            if key in target_data and "r2" in target_data[key]:
                mat[bi, ti] = target_data[key]["r2"]
    return mat


def extract_metric(data, target, blocks, timesteps, metric="r2"):
    """Like extract_heatmap but for an arbitrary metric."""
    n_blocks = len(blocks)
    n_ts = len(timesteps)
    mat = np.full((n_blocks, n_ts), np.nan)
    if target not in data:
        return mat
    target_data = data[target]
    for bi, block in enumerate(blocks):
        for ti, ts in enumerate(timesteps):
            key = f"{block}_{ts}"
            if key in target_data and metric in target_data[key]:
                mat[bi, ti] = target_data[key][metric]
    return mat


def best_r2(data, target, blocks, timesteps):
    """Return the best R2 across all blocks and timesteps."""
    mat = extract_heatmap(data, target, blocks, timesteps)
    if np.all(np.isnan(mat)):
        return np.nan
    return float(np.nanmax(mat))


def best_metric(data, target, blocks, timesteps, metric="balanced_acc"):
    """Return the best value of a metric across all blocks and timesteps."""
    mat = extract_metric(data, target, blocks, timesteps, metric)
    if np.all(np.isnan(mat)):
        return np.nan
    return float(np.nanmax(mat))


def get_object_targets(data):
    """Return sorted list of object target keys (e.g. obj_chair, obj_wall, ...)."""
    return sorted([k for k in data.keys() if k.startswith("obj_")])


def make_block_labels(blocks):
    """Create Y-axis tick labels for blocks."""
    labels = []
    for b in blocks:
        labels.append(b.replace("joint_", "J").replace("single_", "S")
                      .replace("cn_joint_", "CJ").replace("cn_single_", "CS"))
    return labels


def _save(fig, name):
    """Save figure as both PDF and PNG."""
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    fig.savefig(OUTPUT_DIR / f"{name}.png")
    print(f"  Saved: {name}.pdf / .png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Main Results Heatmap
# ---------------------------------------------------------------------------
def fig1_main_heatmap(results):
    """Heatmap of R2 values for the main model across all DiT blocks and timesteps."""
    data = results.get("Main (LoRA)")
    if data is None:
        print("  Skipping Figure 1: Main results not available.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 10), sharey=True)

    block_labels = make_block_labels(DIT_BLOCKS)
    ytick_step = 1
    yticks = np.arange(0, N_DIT, ytick_step)
    yticklabels = [block_labels[i] for i in yticks]

    for ax_i, target in enumerate(CONTINUOUS_TARGETS):
        mat = extract_heatmap(data, target, DIT_BLOCKS, TIMESTEPS)
        im = axes[ax_i].imshow(
            mat, aspect="auto", cmap="viridis", vmin=0, vmax=1,
            interpolation="nearest",
        )
        axes[ax_i].set_title(TARGET_DISPLAY[target], fontweight="bold")
        axes[ax_i].set_xlabel("Timestep")
        axes[ax_i].set_xticks(np.arange(N_TIMESTEPS))
        axes[ax_i].set_xticklabels(TIMESTEP_LABELS)

        # Horizontal line separating joint and single blocks
        axes[ax_i].axhline(y=N_JOINT - 0.5, color="white", linewidth=1.5, linestyle="--")

    axes[0].set_ylabel("Block")
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(yticklabels, fontsize=6)

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("$R^2$")

    fig.suptitle("Main Model (LoRA) -- Linear Probe $R^2$ across DiT Blocks and Timesteps",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    _save(fig, "fig1_main_heatmap")


# ---------------------------------------------------------------------------
# Figure 2: Main vs Control Comparison
# ---------------------------------------------------------------------------
def fig2_main_vs_control(results):
    """Side-by-side heatmaps for Main and Control conditions."""
    main_data = results.get("Main (LoRA)")
    ctrl_data = results.get("Control (no LoRA)")
    if main_data is None or ctrl_data is None:
        print("  Skipping Figure 2: need both Main and Control results.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 14), sharey=True, sharex=True)

    block_labels = make_block_labels(DIT_BLOCKS)
    ytick_step = 1
    yticks = np.arange(0, N_DIT, ytick_step)
    yticklabels = [block_labels[i] for i in yticks]

    conditions = [("Main (LoRA)", main_data), ("Control (no LoRA)", ctrl_data)]

    for row, (cond_label, cond_data) in enumerate(conditions):
        for col, target in enumerate(CONTINUOUS_TARGETS):
            ax = axes[row, col]
            mat = extract_heatmap(cond_data, target, DIT_BLOCKS, TIMESTEPS)
            im = ax.imshow(
                mat, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(TARGET_DISPLAY[target], fontweight="bold")
            ax.set_xticks(np.arange(N_TIMESTEPS))
            ax.set_xticklabels(TIMESTEP_LABELS)
            ax.axhline(y=N_JOINT - 0.5, color="white", linewidth=1.5, linestyle="--")

            if col == 0:
                ax.set_ylabel(cond_label, fontweight="bold")
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels, fontsize=6)

    for ax in axes[-1]:
        ax.set_xlabel("Timestep")

    cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label("$R^2$")

    fig.suptitle("Main vs Control -- Linear Probe $R^2$",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 0.95, 0.97])
    _save(fig, "fig2_main_vs_control")


# ---------------------------------------------------------------------------
# Figure 3: LoRA Delta Heatmap
# ---------------------------------------------------------------------------
def fig3_delta_heatmap(results):
    """Heatmap of R2(Main) - R2(Control) for each target."""
    main_data = results.get("Main (LoRA)")
    ctrl_data = results.get("Control (no LoRA)")
    if main_data is None or ctrl_data is None:
        print("  Skipping Figure 3: need both Main and Control results.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 10), sharey=True)

    block_labels = make_block_labels(DIT_BLOCKS)
    ytick_step = 1
    yticks = np.arange(0, N_DIT, ytick_step)
    yticklabels = [block_labels[i] for i in yticks]

    for ax_i, target in enumerate(CONTINUOUS_TARGETS):
        mat_main = extract_heatmap(main_data, target, DIT_BLOCKS, TIMESTEPS)
        mat_ctrl = extract_heatmap(ctrl_data, target, DIT_BLOCKS, TIMESTEPS)
        delta = mat_main - mat_ctrl

        # Find symmetric range
        vmax = np.nanmax(np.abs(delta))
        if vmax == 0 or np.isnan(vmax):
            vmax = 0.1
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = axes[ax_i].imshow(
            delta, aspect="auto", cmap="RdBu_r", norm=norm,
            interpolation="nearest",
        )
        axes[ax_i].set_title(f"$\\Delta$ {TARGET_DISPLAY[target]}", fontweight="bold")
        axes[ax_i].set_xlabel("Timestep")
        axes[ax_i].set_xticks(np.arange(N_TIMESTEPS))
        axes[ax_i].set_xticklabels(TIMESTEP_LABELS)
        axes[ax_i].axhline(y=N_JOINT - 0.5, color="black", linewidth=1, linestyle="--")

        cbar_i = fig.colorbar(im, ax=axes[ax_i], shrink=0.6, pad=0.02)
        cbar_i.set_label("$\\Delta R^2$ (Main $-$ Control)")

    axes[0].set_ylabel("Block")
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(yticklabels, fontsize=6)

    fig.suptitle("LoRA Effect: $R^2$(Main) $-$ $R^2$(Control)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig3_delta_heatmap")


# ---------------------------------------------------------------------------
# Figure 4: Best-timestep Line Plot
# ---------------------------------------------------------------------------
def fig4_best_timestep_lineplot(results):
    """For each target, pick the best timestep and plot R2 vs block depth."""
    main_data = results.get("Main (LoRA)")
    ctrl_data = results.get("Control (no LoRA)")
    if main_data is None:
        print("  Skipping Figure 4: Main results not available.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    conditions = {"Main (LoRA)": main_data}
    if ctrl_data is not None:
        conditions["Control (no LoRA)"] = ctrl_data

    colors = {"Main (LoRA)": "#1f77b4", "Control (no LoRA)": "#ff7f0e"}

    for ax_i, target in enumerate(CONTINUOUS_TARGETS):
        for cond_label, cond_data in conditions.items():
            mat = extract_heatmap(cond_data, target, DIT_BLOCKS, TIMESTEPS)
            # Find the timestep with the best average R2 across blocks
            mean_per_ts = np.nanmean(mat, axis=0)
            best_ts_idx = int(np.nanargmax(mean_per_ts))
            r2_at_best_ts = mat[:, best_ts_idx]

            axes[ax_i].plot(
                np.arange(N_DIT), r2_at_best_ts,
                label=f"{cond_label} (t={TIMESTEP_LABELS[best_ts_idx]})",
                color=colors[cond_label], linewidth=1.5, alpha=0.9,
            )

        axes[ax_i].set_title(TARGET_DISPLAY[target], fontweight="bold")
        axes[ax_i].set_xlabel("Block Index")
        axes[ax_i].axvline(x=N_JOINT - 0.5, color="gray", linewidth=0.8,
                           linestyle="--", alpha=0.6)
        axes[ax_i].text(N_JOINT / 2, -0.05, "Joint", ha="center", fontsize=8,
                        color="gray", transform=axes[ax_i].get_xaxis_transform())
        axes[ax_i].text(N_JOINT + N_SINGLE / 2, -0.05, "Single", ha="center",
                        fontsize=8, color="gray",
                        transform=axes[ax_i].get_xaxis_transform())
        axes[ax_i].legend(frameon=False, loc="lower right")

    axes[0].set_ylabel("$R^2$")

    fig.suptitle("Information Flow: $R^2$ vs Block Depth (Best Timestep)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig4_best_timestep_lineplot")


# ---------------------------------------------------------------------------
# Figure 5: DiT vs ControlNet Comparison
# ---------------------------------------------------------------------------
def fig5_dit_vs_cn(results):
    """Bar chart comparing best R2 from DiT global vs CN global for each target."""
    conditions_available = {k: v for k, v in results.items() if v is not None}
    if not conditions_available:
        print("  Skipping Figure 5: no results available.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

    cond_labels = list(conditions_available.keys())
    n_conds = len(cond_labels)

    bar_width = 0.35
    for ax_i, target in enumerate(CONTINUOUS_TARGETS):
        dit_target = target
        cn_target = f"cn_{target}"

        x = np.arange(n_conds)
        dit_vals = []
        cn_vals = []
        for cl in cond_labels:
            d = conditions_available[cl]
            dit_vals.append(best_r2(d, dit_target, DIT_BLOCKS, TIMESTEPS))
            cn_vals.append(best_r2(d, cn_target, CN_BLOCKS, TIMESTEPS))

        bars1 = axes[ax_i].bar(x - bar_width / 2, dit_vals, bar_width,
                               label="DiT", color="#4c72b0", edgecolor="white")
        bars2 = axes[ax_i].bar(x + bar_width / 2, cn_vals, bar_width,
                               label="ControlNet", color="#dd8452", edgecolor="white")

        axes[ax_i].set_title(TARGET_DISPLAY[target], fontweight="bold")
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(cond_labels, fontsize=8, rotation=15, ha="right")
        axes[ax_i].set_ylim(0, 1.08)

        # Add value labels on bars
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if not np.isnan(h):
                axes[ax_i].text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                                f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    axes[0].set_ylabel("Best $R^2$")
    axes[0].legend(frameon=False)

    fig.suptitle("DiT vs ControlNet: Best Linear Probe $R^2$",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig5_dit_vs_cn")


# ---------------------------------------------------------------------------
# Figure 6: Global vs Spatial Comparison
# ---------------------------------------------------------------------------
def fig6_global_vs_spatial(results):
    """Grouped bar chart: DiT global, DiT spatial, CN global for each target."""
    conditions_available = {k: v for k, v in results.items() if v is not None}
    if not conditions_available:
        print("  Skipping Figure 6: no results available.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    cond_labels = list(conditions_available.keys())
    n_conds = len(cond_labels)
    probe_colors = ["#4c72b0", "#55a868", "#dd8452"]

    bar_width = 0.22

    for ax_i, target in enumerate(CONTINUOUS_TARGETS):
        dit_target = target
        spatial_target = f"spatial_{target}"
        cn_target = f"cn_{target}"

        x = np.arange(n_conds)
        for pi, (ptype, ptarget, pblocks) in enumerate([
            ("DiT Global", dit_target, DIT_BLOCKS),
            ("DiT Spatial", spatial_target, SPATIAL_DIT_BLOCKS),
            ("CN Global", cn_target, CN_BLOCKS),
        ]):
            vals = []
            for cl in cond_labels:
                d = conditions_available[cl]
                vals.append(best_r2(d, ptarget, pblocks, TIMESTEPS))

            bars = axes[ax_i].bar(x + (pi - 1) * bar_width, vals, bar_width,
                                  label=ptype if ax_i == 0 else None,
                                  color=probe_colors[pi], edgecolor="white")
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    axes[ax_i].text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                                    f"{h:.3f}", ha="center", va="bottom", fontsize=6)

        axes[ax_i].set_title(TARGET_DISPLAY[target], fontweight="bold")
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(cond_labels, fontsize=8, rotation=15, ha="right")
        axes[ax_i].set_ylim(0, 1.12)

    axes[0].set_ylabel("Best $R^2$")
    axes[0].legend(frameon=False, loc="lower right")

    fig.suptitle("Global vs Spatial Probing: Best $R^2$",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig6_global_vs_spatial")


# ---------------------------------------------------------------------------
# Figure 7: Object Presence Probing
# ---------------------------------------------------------------------------
def fig7_object_probing(results):
    """Horizontal bar chart showing best balanced_acc per object class."""
    main_data = results.get("Main (LoRA)")
    ctrl_data = results.get("Control (no LoRA)")
    if main_data is None:
        print("  Skipping Figure 7: Main results not available.")
        return

    # Find object targets from main results
    obj_targets = get_object_targets(main_data)
    if not obj_targets:
        print("  Skipping Figure 7: no object targets found.")
        return

    obj_names = [t.replace("obj_", "").replace("_", " ").title() for t in obj_targets]

    # Get best balanced_acc for each object (using DiT blocks)
    main_vals = []
    ctrl_vals = []
    for t in obj_targets:
        main_vals.append(best_metric(main_data, t, DIT_BLOCKS, TIMESTEPS, "balanced_acc"))
        if ctrl_data is not None:
            ctrl_vals.append(best_metric(ctrl_data, t, DIT_BLOCKS, TIMESTEPS, "balanced_acc"))

    main_vals = np.array(main_vals)
    ctrl_vals = np.array(ctrl_vals) if ctrl_vals else None

    # Sort by Main balanced_acc (descending)
    sort_idx = np.argsort(main_vals)[::-1]
    obj_names = [obj_names[i] for i in sort_idx]
    obj_targets_sorted = [obj_targets[i] for i in sort_idx]
    main_vals = main_vals[sort_idx]
    if ctrl_vals is not None:
        ctrl_vals = ctrl_vals[sort_idx]

    n_obj = len(obj_names)
    y = np.arange(n_obj)
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(8, max(6, n_obj * 0.35)))

    bars_main = ax.barh(y - bar_h / 2, main_vals, bar_h,
                        label="Main (LoRA)", color="#4c72b0", edgecolor="white")
    if ctrl_vals is not None:
        bars_ctrl = ax.barh(y + bar_h / 2, ctrl_vals, bar_h,
                            label="Control (no LoRA)", color="#ff7f0e", edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(obj_names, fontsize=9)
    ax.set_xlabel("Best Balanced Accuracy")
    ax.set_xlim(0.4, 1.02)
    ax.axvline(x=0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()

    fig.suptitle("Object Presence: Best Balanced Accuracy per Class",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig7_object_probing")

    return obj_targets_sorted, main_vals, ctrl_vals


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------
def print_summary(results):
    """Print a comprehensive summary of key metrics."""
    print("\n" + "=" * 80)
    print("PROBING RESULTS SUMMARY")
    print("=" * 80)

    conditions_available = {k: v for k, v in results.items() if v is not None}

    # --- Continuous targets: best R2 ---
    print("\n--- Best R2 for Continuous Targets (DiT Global) ---")
    header = f"{'Target':<15}" + "".join(f"{c:<25}" for c in conditions_available)
    print(header)
    print("-" * len(header))
    for target in CONTINUOUS_TARGETS:
        row = f"{TARGET_DISPLAY[target]:<15}"
        for cl, d in conditions_available.items():
            mat = extract_heatmap(d, target, DIT_BLOCKS, TIMESTEPS)
            best = np.nanmax(mat) if not np.all(np.isnan(mat)) else np.nan
            # Find best block, timestep
            if not np.isnan(best):
                idx = np.unravel_index(np.nanargmax(mat), mat.shape)
                block_name = DIT_BLOCKS[idx[0]]
                ts_name = TIMESTEPS[idx[1]]
                cell = f"{best:.4f} ({block_name}_{ts_name})"
                row += f"{cell:<25}"
            else:
                row += f"{'N/A':<25}"
        print(row)

    # --- CN targets ---
    print("\n--- Best R2 for Continuous Targets (ControlNet Global) ---")
    header = f"{'Target':<15}" + "".join(f"{c:<25}" for c in conditions_available)
    print(header)
    print("-" * len(header))
    for target in CONTINUOUS_TARGETS:
        cn_target = f"cn_{target}"
        row = f"{TARGET_DISPLAY[target]:<15}"
        for cl, d in conditions_available.items():
            mat = extract_heatmap(d, cn_target, CN_BLOCKS, TIMESTEPS)
            best = np.nanmax(mat) if not np.all(np.isnan(mat)) else np.nan
            if not np.isnan(best):
                idx = np.unravel_index(np.nanargmax(mat), mat.shape)
                block_name = CN_BLOCKS[idx[0]]
                ts_name = TIMESTEPS[idx[1]]
                cell = f"{best:.4f} ({block_name}_{ts_name})"
                row += f"{cell:<25}"
            else:
                row += f"{'N/A':<25}"
        print(row)

    # --- Spatial targets ---
    print("\n--- Best R2 for Continuous Targets (DiT Spatial) ---")
    header = f"{'Target':<15}" + "".join(f"{c:<25}" for c in conditions_available)
    print(header)
    print("-" * len(header))
    for target in CONTINUOUS_TARGETS:
        sp_target = f"spatial_{target}"
        row = f"{TARGET_DISPLAY[target]:<15}"
        for cl, d in conditions_available.items():
            mat = extract_heatmap(d, sp_target, SPATIAL_DIT_BLOCKS, TIMESTEPS)
            best = np.nanmax(mat) if not np.all(np.isnan(mat)) else np.nan
            if not np.isnan(best):
                idx = np.unravel_index(np.nanargmax(mat), mat.shape)
                block_name = SPATIAL_DIT_BLOCKS[idx[0]]
                ts_name = TIMESTEPS[idx[1]]
                cell = f"{best:.4f} ({block_name}_{ts_name})"
                row += f"{cell:<25}"
            else:
                row += f"{'N/A':<25}"
        print(row)

    # --- LoRA delta for continuous targets ---
    main_data = results.get("Main (LoRA)")
    ctrl_data = results.get("Control (no LoRA)")
    if main_data is not None and ctrl_data is not None:
        print("\n--- LoRA Delta (Main - Control) for DiT Global ---")
        for target in CONTINUOUS_TARGETS:
            mat_main = extract_heatmap(main_data, target, DIT_BLOCKS, TIMESTEPS)
            mat_ctrl = extract_heatmap(ctrl_data, target, DIT_BLOCKS, TIMESTEPS)
            delta = mat_main - mat_ctrl
            avg_delta = np.nanmean(delta)
            max_delta = np.nanmax(delta)
            min_delta = np.nanmin(delta)
            print(f"  {TARGET_DISPLAY[target]:<15}: "
                  f"mean={avg_delta:+.4f}, max={max_delta:+.4f}, min={min_delta:+.4f}")

    # --- Object probing summary ---
    if main_data is not None:
        obj_targets = get_object_targets(main_data)
        if obj_targets:
            print(f"\n--- Object Probing (Best Balanced Accuracy, DiT) ---")
            print(f"  {'Object':<25} {'Main':<12} {'Control':<12} {'Delta':<12}")
            print("  " + "-" * 60)
            for t in sorted(obj_targets):
                name = t.replace("obj_", "").replace("_", " ").title()
                main_val = best_metric(main_data, t, DIT_BLOCKS, TIMESTEPS, "balanced_acc")
                ctrl_val = (best_metric(ctrl_data, t, DIT_BLOCKS, TIMESTEPS, "balanced_acc")
                            if ctrl_data else np.nan)
                delta_str = ""
                if not np.isnan(main_val) and not np.isnan(ctrl_val):
                    delta_str = f"{main_val - ctrl_val:+.4f}"
                ctrl_str = f"{ctrl_val:.4f}" if not np.isnan(ctrl_val) else "N/A"
                print(f"  {name:<25} {main_val:<12.4f} {ctrl_str:<12} {delta_str}")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("SPAD Probing Analysis")
    print("=" * 60)

    print("\nLoading results...")
    results = load_results()
    if not any(v is not None for v in results.values()):
        print("ERROR: No result files found. Exiting.")
        sys.exit(1)

    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    print("Generating Figure 1: Main Results Heatmap...")
    fig1_main_heatmap(results)

    print("Generating Figure 2: Main vs Control Comparison...")
    fig2_main_vs_control(results)

    print("Generating Figure 3: LoRA Delta Heatmap...")
    fig3_delta_heatmap(results)

    print("Generating Figure 4: Best-timestep Line Plot...")
    fig4_best_timestep_lineplot(results)

    print("Generating Figure 5: DiT vs ControlNet Comparison...")
    fig5_dit_vs_cn(results)

    print("Generating Figure 6: Global vs Spatial Comparison...")
    fig6_global_vs_spatial(results)

    print("Generating Figure 7: Object Presence Probing...")
    fig7_object_probing(results)

    print_summary(results)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
