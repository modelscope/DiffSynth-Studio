#!/usr/bin/env python3
"""
Photon-flux calibration figure.

For each SPAD filter (RAW_empty, RAW_OD_01, RAW_OD_03, RAW_OD_07), compute the
median per-pixel detected photon count per frame from real captures, convert
to incident photons-per-pixel-per-second using sensor PDE and frame rate, and
plot on a log axis alongside published low-light CMOS benchmarks.

Papers compared:
  • SID — Chen et al., CVPR 2018. Indoor 0.03–0.3 lux, outdoor 0.2–5 lux,
        Sony A7S II / Fuji X-T2. Pixel pitch 5.95 µm (Sony) → photopic
        conversion ≈ 1.5e3 photons/pixel/sec/lux.
  • ELD — Wei et al., CVPR 2020. Noise-model regime is 100–300 photons per
        long exposure (~10 s), → 10–30 photons/pixel/sec.
  • Starlight — Monakhova et al., CVPR 2022. Demonstrates video at
        0.6–0.7 millilux (no moon), down to <0.001 lux.  ≈ 1 photon/pixel/sec.
  • Quanta Burst Photography — Ma et al., SIGGRAPH 2020. SPAD, <0.06
        photons/pixel/frame at SwissSPAD2 ~17 kHz → ~1000 photons/pixel/sec.
  • bit2bit — Liu et al., NeurIPS 2024. Same regime as QBP, <0.06
        photons/pixel/frame.

Key claim of the figure: with OD7, our SPAD captures sit 1–3 orders of
magnitude below all CMOS benchmarks, and at or below the SPAD benchmarks.
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
from spad_utils_fixed import accumulate_counts_whole_file

H, W = 512, 512
BYTES_PER_FRAME = (H * W) // 8

# Our SwissSPAD2 sensor (see /nfs/horai.dgpsrv/.../THESIS_CONTEXT and similar):
SPAD_FRAME_RATE_HZ = 16_667.0  # ~60 µs/frame; SwissSPAD2 nominal
SPAD_PDE = 0.05                 # conservative visible-band PDE; SwissSPAD2 5–10 %
SPAD_PIXEL_PITCH_UM = 16.38     # SwissSPAD2 pitch
SPAD_FILL_FACTOR = 0.10         # native fill factor (no microlens)

# Published CMOS conversion (photopic, 555 nm equivalent):
LUX_TO_PHOTONS_PER_M2_PER_SEC = 4.08e15  # at 555 nm peak
# = 1 lux × (1 lm / 683 W) × (1 / (h c/λ at 555 nm))


def lux_to_photons_per_pixel_per_sec(lux, pixel_pitch_um):
    """Photons incident on a single pixel per second at the given illuminance.

    Assumes 555-nm-equivalent photopic conversion. This is an order-of-
    magnitude estimate; actual value depends on scene spectrum.
    """
    A_m2 = (pixel_pitch_um * 1e-6) ** 2
    return lux * A_m2 * LUX_TO_PHOTONS_PER_M2_PER_SEC


# Published benchmarks. Each entry:
#   (label, low_photons_per_pixel_per_sec, high_photons_per_pixel_per_sec,
#    color, sensor_label)
# Conversion details in comments.
BENCHMARKS = [
    # SID: 0.03–5 lux, Sony A7S2 (5.95 µm)
    {
        "label": "SID (Chen 2018)\nSony A7S II",
        "low":  lux_to_photons_per_pixel_per_sec(0.03, 5.95),
        "high": lux_to_photons_per_pixel_per_sec(5.0,  5.95),
        "color": "tab:blue",
        "type": "CMOS",
    },
    # ELD: 100–300 photons per long exposure (~10 s) → 10–30/sec at sensor
    {
        "label": "ELD (Wei 2020)\nSony A7S2 / Nikon / Canon",
        "low":  10.0,
        "high": 30.0,
        "color": "tab:cyan",
        "type": "CMOS",
    },
    # Starlight: 0.6–0.7 mlux down to <0.001 lux, Sony IMX291 / similar low-light CMOS (2.9 µm)
    {
        "label": "Starlight (Monakhova 2022)\nSony IMX291",
        "low":  lux_to_photons_per_pixel_per_sec(0.0001, 2.9),
        "high": lux_to_photons_per_pixel_per_sec(0.001,  2.9),
        "color": "tab:purple",
        "type": "CMOS",
    },
    # QBP: <0.06 detected/frame, SwissSPAD2 17 kHz → <1000 detected/sec.
    # Convert to incident: divide by SPAD_PDE → up to 20 000 incident/sec
    {
        "label": "Quanta Burst (Ma 2020)\nSwissSPAD2",
        "low":  0.001 * SPAD_FRAME_RATE_HZ / SPAD_PDE,
        "high": 0.06  * SPAD_FRAME_RATE_HZ / SPAD_PDE,
        "color": "tab:orange",
        "type": "SPAD",
    },
    # bit2bit: same regime as QBP
    {
        "label": "bit2bit (Liu 2024)\nSwissSPAD2",
        "low":  0.001 * SPAD_FRAME_RATE_HZ / SPAD_PDE,
        "high": 0.06  * SPAD_FRAME_RATE_HZ / SPAD_PDE,
        "color": "tab:red",
        "type": "SPAD",
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="/nfs/horai.dgpsrv/ondemand30/jw954/images")
    p.add_argument("--output-dir", default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
    p.add_argument("--n-scenes", type=int, default=200,
                   help="Number of scenes to sample (median across scenes)")
    p.add_argument("--n-frames", type=int, default=1000,
                   help="Frames per file to accumulate for rate estimate")
    p.add_argument("--frame-rate", type=float, default=SPAD_FRAME_RATE_HZ,
                   help="SPAD frame rate in Hz (default SwissSPAD2 16.67 kHz)")
    p.add_argument("--pde", type=float, default=SPAD_PDE,
                   help="SPAD photon detection efficiency (default 0.05)")
    return p.parse_args()


def measure_filter_flux(images_dir, filter_name, scenes, n_frames):
    """For a given filter (e.g. RAW_empty), accumulate per-scene Bernoulli rates
    over the first n_frames frames, and return the per-pixel mean detected
    photons per frame across all (scene, pixel) tuples that aren't clipped.

    Returns: (per_scene_lambda_array, per_scene_mean_p, n_scenes_used)
    """
    lam_per_scene = []
    p_per_scene = []
    for scene in scenes:
        bin_path = Path(images_dir) / scene / f"{filter_name}.bin"
        if not bin_path.exists():
            continue
        try:
            bytes_needed = n_frames * BYTES_PER_FRAME
            with open(bin_path, "rb") as f:
                raw = np.frombuffer(f.read(bytes_needed), dtype=np.uint8)
            counts, n_actual = accumulate_counts_whole_file(raw, n_frames, H, W)
        except Exception:
            continue
        if n_actual < 100:
            continue
        p = counts.astype(np.float32) / n_actual
        # Use mid-range pixels only (avoid clipped 0 or near-1)
        mid = (p > 0.005) & (p < 0.95)
        if mid.sum() < 1000:
            continue
        # Per-pixel detected photons per frame
        lam = -np.log1p(-np.clip(p[mid], 0.0, 1.0 - 1e-6))
        lam_per_scene.append(float(np.median(lam)))
        p_per_scene.append(float(np.median(p[mid])))
    return np.array(lam_per_scene), np.array(p_per_scene)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    images_dir = Path(args.images_dir)
    all_scenes = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    # Sample evenly across the dataset for diversity
    step = max(1, len(all_scenes) // args.n_scenes)
    scenes = all_scenes[::step][: args.n_scenes]
    print(f"Total scenes available: {len(all_scenes)}, sampling {len(scenes)} (every {step}-th)")

    print(f"\nSPAD model: PDE={args.pde:.2%}, frame_rate={args.frame_rate/1000:.2f} kHz")
    print(f"  → 1 detected photon/frame ≈ {1.0 / args.pde:.1f} incident photons absorbed")
    print(f"  → photons/sec = photons/frame × {args.frame_rate:.0f}")
    print()

    filters = ["RAW_empty", "RAW_OD_01", "RAW_OD_03", "RAW_OD_07"]
    transmission = {"RAW_empty": 1.0, "RAW_OD_01": 10**-0.1,
                    "RAW_OD_03": 10**-0.3, "RAW_OD_07": 10**-0.7}

    # Measure each filter
    spad_results = {}
    print(f"{'filter':<12}  {'n_scenes':>9}  {'median p̂':>10}  {'λ_det/frame':>12}  "
          f"{'λ_inc/frame':>12}  {'photons/sec (incident)':>24}")
    print("-" * 90)
    for filt in filters:
        t0 = time.time()
        lam_det_per_scene, p_med_per_scene = measure_filter_flux(
            images_dir, filt, scenes, args.n_frames
        )
        if len(lam_det_per_scene) == 0:
            print(f"  {filt:<12}: NO DATA")
            continue
        # Median across scenes (robust to scene-content variation)
        lam_det = float(np.median(lam_det_per_scene))
        lam_inc_per_frame = lam_det / args.pde
        photons_per_sec = lam_inc_per_frame * args.frame_rate
        spad_results[filt] = {
            "lam_det_per_frame": lam_det,
            "lam_inc_per_frame": lam_inc_per_frame,
            "photons_per_sec":  photons_per_sec,
            "n_scenes":         len(lam_det_per_scene),
            "p_median":         float(np.median(p_med_per_scene)),
            "p_per_scene":      p_med_per_scene,
            "lam_det_per_scene": lam_det_per_scene,
            "transmission":     transmission[filt],
            "elapsed_s":        time.time() - t0,
        }
        print(f"  {filt:<12}  {len(lam_det_per_scene):>9}  "
              f"{spad_results[filt]['p_median']:>10.4f}  "
              f"{lam_det:>12.4f}  {lam_inc_per_frame:>12.4f}  "
              f"{photons_per_sec:>24.2f}  ({time.time()-t0:.1f}s)")

    # Save numbers for downstream use
    out_npz = out_dir / "photon_flux_results.npz"
    save = {
        "frame_rate_hz": args.frame_rate,
        "pde": args.pde,
        "spad_pixel_pitch_um": SPAD_PIXEL_PITCH_UM,
        "lux_to_photons_per_m2_per_sec": LUX_TO_PHOTONS_PER_M2_PER_SEC,
    }
    for filt, r in spad_results.items():
        save[f"{filt}_p_median"] = r["p_median"]
        save[f"{filt}_lam_det_per_frame"] = r["lam_det_per_frame"]
        save[f"{filt}_lam_inc_per_frame"] = r["lam_inc_per_frame"]
        save[f"{filt}_photons_per_sec"] = r["photons_per_sec"]
        save[f"{filt}_n_scenes"] = r["n_scenes"]
        save[f"{filt}_transmission"] = r["transmission"]
    np.savez(out_npz, **save)
    print(f"\nResults saved → {out_npz}")

    # =======================================================================
    # Build the comparison figure
    # =======================================================================
    print("\nBuilding comparison figure …")
    fig, ax = plt.subplots(figsize=(11, 7))

    y_pos = 0
    yticks = []
    yticklabels = []

    # CMOS benchmarks (top group, blue family)
    for b in BENCHMARKS:
        if b["type"] != "CMOS":
            continue
        ax.plot([b["low"], b["high"]], [y_pos, y_pos],
                "-", lw=8, alpha=0.7, color=b["color"], solid_capstyle="round")
        yticks.append(y_pos)
        yticklabels.append(b["label"])
        y_pos += 1

    # Separator
    ax.axhline(y_pos - 0.5, color="gray", ls=":", lw=0.8, alpha=0.5)

    # SPAD benchmarks
    for b in BENCHMARKS:
        if b["type"] != "SPAD":
            continue
        ax.plot([b["low"], b["high"]], [y_pos, y_pos],
                "-", lw=8, alpha=0.7, color=b["color"], solid_capstyle="round")
        yticks.append(y_pos)
        yticklabels.append(b["label"])
        y_pos += 1

    # Separator
    ax.axhline(y_pos - 0.5, color="gray", ls=":", lw=0.8, alpha=0.5)

    # Our SPAD with each filter — show the DARKEST scenes too (P1 of medians)
    spad_colors = {"RAW_empty": "darkgreen", "RAW_OD_01": "olive",
                   "RAW_OD_03": "darkorange", "RAW_OD_07": "darkred"}
    nice_label = {"RAW_empty": "Ours, no filter (RAW_empty)",
                  "RAW_OD_01": "Ours, OD 0.1 (79% T)",
                  "RAW_OD_03": "Ours, OD 0.3 (50% T)",
                  "RAW_OD_07": "Ours, OD 0.7 (20% T)  ★"}
    for filt in filters:
        if filt not in spad_results:
            continue
        r = spad_results[filt]
        lams_per_scene = r["lam_det_per_scene"] / args.pde * args.frame_rate
        p1, p5, mid, p95, p99 = np.percentile(lams_per_scene, [1, 5, 50, 95, 99])
        # Wider full-range bar (P1–P99) in light, inner P5–P95 in dark
        ax.plot([p1, p99], [y_pos, y_pos], "-", lw=4, alpha=0.4,
                color=spad_colors[filt], solid_capstyle="round")
        ax.plot([p5, p95], [y_pos, y_pos], "-", lw=8, alpha=0.85,
                color=spad_colors[filt], solid_capstyle="round")
        ax.plot([mid], [y_pos], "k|", ms=14, mew=2)
        # Mark dimmest individual scene
        dimmest = lams_per_scene.min()
        ax.plot([dimmest], [y_pos], "v", color=spad_colors[filt], ms=10,
                mec="black", mew=1)
        yticks.append(y_pos)
        yticklabels.append(f"{nice_label[filt]}\n({r['n_scenes']} scenes, ▼=dimmest)")
        y_pos += 1

    ax.set_xscale("log")
    ax.set_xlabel("Incident photon flux at sensor  (photons / pixel / second)",
                  fontsize=12)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=10)
    ax.set_xlim(0.05, 1e6)
    ax.grid(True, axis="x", which="both", alpha=0.3)
    ax.invert_yaxis()
    ax.set_title(
        "Where does our SPAD dataset sit in low-light imaging?\n"
        "Bars: thick = P5–P95, thin = P1–P99. │ = median, ▼ = dimmest scene.\n"
        "OD 0.7 is the darkest end of our dataset; reaches into Starlight/ELD regime.",
        fontsize=12,
    )

    # Vertical guides at key flux levels
    for x, lbl in [(1.0, "1 ph/sec"), (10.0, "10"), (1e2, "100"),
                   (1e3, "1k"), (1e4, "10k"), (1e5, "100k")]:
        ax.axvline(x, color="gray", ls=":", lw=0.5, alpha=0.4)

    fig.tight_layout()
    out_pdf = fig_dir / "photon_flux_calibration.pdf"
    out_png = fig_dir / "photon_flux_calibration.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")

    # Summary print
    print("\n" + "=" * 70)
    print("  SUMMARY (incident photons / pixel / second)")
    print("=" * 70)
    for b in BENCHMARKS:
        print(f"  {b['label'].split(chr(10))[0]:<35}  "
              f"{b['low']:.2g} – {b['high']:.2g}")
    print()
    for filt in filters:
        if filt in spad_results:
            r = spad_results[filt]
            lams_per_scene = r["lam_det_per_scene"] / args.pde * args.frame_rate
            lo, mid, hi = np.percentile(lams_per_scene, [5, 50, 95])
            print(f"  Ours {filt:<10}  {lo:.2g} (P5) – {mid:.2g} (median) – {hi:.2g} (P95)")
    print()
    if "RAW_OD_07" in spad_results and "RAW_empty" in spad_results:
        ratio = spad_results["RAW_empty"]["photons_per_sec"] / spad_results["RAW_OD_07"]["photons_per_sec"]
        print(f"  Empty / OD7 ratio: {ratio:.1f}× (expected ~5× from OD 0.7 = 0.20 transmission)")


if __name__ == "__main__":
    main()
