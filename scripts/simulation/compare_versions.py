#!/usr/bin/env python3
"""
Compare calibration versions v1, v3, v4 side-by-side.

Loads each NPZ, prints a unified table, generates a combined diagnostic figure.

Run after v1, v3, v4 have completed.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAL_DIR = Path("/nfs/horai.dgpsrv/ondemand30/jw954/calibration")

VERSIONS = {
    "v1 (raw p-space)":   "rgb_to_mono_weights.npz",
    "v3 (λ-space + hpfix)": "rgb_to_mono_weights_v3.npz",
    "v4 (processed PNGs)": "rgb_to_mono_weights_v4.npz",
}

# -------- Load --------
results = {}
for name, fn in VERSIONS.items():
    path = CAL_DIR / fn
    if not path.exists():
        print(f"[warn] missing: {path}")
        continue
    d = np.load(path)
    results[name] = {
        "w_r": float(d["w_r"]), "w_g": float(d["w_g"]), "w_b": float(d["w_b"]),
        "sum": float(np.array(d["weights"]).sum()),
        "train_r2": float(d["train_r2"]), "val_r2": float(d["val_r2"]),
        "train_N": int(d["train_N"]), "val_N": int(d["val_N"]),
    }

# -------- Print table --------
print()
print("=" * 84)
print("  Calibration Version Comparison")
print("=" * 84)
header = f"{'Version':<26}  {'w_r':>8}  {'w_g':>8}  {'w_b':>8}  {'sum':>7}  {'Train R²':>9}  {'Val R²':>8}"
print(header)
print("-" * len(header))
for name, r in results.items():
    print(f"{name:<26}  {r['w_r']:>8.4f}  {r['w_g']:>8.4f}  {r['w_b']:>8.4f}  {r['sum']:>7.3f}  {r['train_r2']:>9.4f}  {r['val_r2']:>8.4f}")
print()

# -------- Pixel counts --------
print(f"{'Version':<26}  {'Train px':>15}  {'Val px':>15}")
print("-" * 60)
for name, r in results.items():
    print(f"{name:<26}  {r['train_N']:>15,}  {r['val_N']:>15,}")
print()

# -------- Combined figure --------
n = len(results)
fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
if n == 1:
    axes = [axes]

colors = {"v1 (raw p-space)": "steelblue",
          "v3 (λ-space + hpfix)": "darkorange",
          "v4 (processed PNGs)": "forestgreen"}

# Reload raw scatter data from the npz files if present
for ax, (name, r) in zip(axes, results.items()):
    fn = VERSIONS[name]
    d = np.load(CAL_DIR / fn)
    # v3 saves λ-space scatter; v1 and v4 save p/RGB-space scatter implicitly via the PNGs
    # Rather than parsing from PNGs, just show summary stats
    ax.axis("off")
    txt = [
        f"{name}",
        "",
        f"w_r = {r['w_r']:+.4f}",
        f"w_g = {r['w_g']:+.4f}",
        f"w_b = {r['w_b']:+.4f}",
        f"sum = {r['sum']:+.4f}",
        "",
        f"Train R² = {r['train_r2']:.4f}",
        f"Val   R² = {r['val_r2']:.4f}",
        "",
        f"Train N = {r['train_N']:,}",
        f"Val   N = {r['val_N']:,}",
    ]
    ax.text(0.05, 0.95, "\n".join(txt), transform=ax.transAxes,
            fontsize=13, family="monospace", verticalalignment="top",
            color=colors.get(name, "black"))

fig.suptitle("Calibration Version Comparison", fontsize=14)
fig.tight_layout()
out = CAL_DIR / "version_comparison_summary.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Summary figure: {out}")

# -------- Save combined NPZ --------
out_npz = CAL_DIR / "calibration_comparison.npz"
save_dict = {}
for name, r in results.items():
    key = name.split()[0]  # "v1", "v3", "v4"
    save_dict[f"{key}_weights"] = np.array([r["w_r"], r["w_g"], r["w_b"]])
    save_dict[f"{key}_train_r2"] = r["train_r2"]
    save_dict[f"{key}_val_r2"] = r["val_r2"]
np.savez(out_npz, **save_dict)
print(f"Combined NPZ:   {out_npz}")
