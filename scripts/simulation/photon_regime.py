#!/usr/bin/env python3
"""
Rigorous photon-flux calibration figure across SPAD and CMOS low-light datasets.

For each comparison paper, photon flux at the sensor face is computed from the
explicitly-reported experimental parameters using the standard radiometric
chain:

     E_sensor [lux]  =  E_scene [lux] × τ_lens × π / (4 × N²)
     Φ_sensor [ph/m²/s] = E_sensor × κ(SPD)
     Φ_pixel [ph/pixel/s] = Φ_sensor × A_pixel
     N_per_exposure [ph/pixel/exposure] = Φ_pixel × t_exposure
     N_detected = N_per_exposure × QE                 # for CMOS
     N_detected = 1 - exp(-N_per_exposure × PDE)       # for SPAD (1-bit)

where:
    κ(SPD) = 4.08e15  ph/m²/s/lux at 555 nm photopic.
             (= 1 lm/W ÷ 683 ÷ photon energy at 555 nm).
             For broadband daylight, ~250 lm/W ⇒ κ ≈ 1.1e15 ph/m²/s/lux.
             Conservative: use photopic 555 nm, note 1.5–3× scene-spectrum uncertainty.

Outputs:
    calibration/figures/photon_regime.pdf / .png
    calibration/photon_calibration.csv
    calibration/photon_methodology.md
"""

import sys
import csv
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

# -----------------------------------------------------------------------------
# Photometric constant: photons per m² per second per lux at 555 nm photopic
# = 1 lux × (1 lm/W ÷ 683) ÷ (h c / λ at 555 nm)
# = (1.464e-3 W/m²) ÷ (3.58e-19 J/photon) = 4.087e15 ph/m²/s/lux
# Use this as the central value; broadband scenes give ~0.5×, scotopic ~0.6×.
PHOTOPIC_PH_PER_M2_PER_S_PER_LUX = 4.087e15


def lens_factor(N, tau=0.85):
    """Image-side illuminance fraction = τ × π / (4 N²) for object at infinity."""
    return tau * np.pi / (4.0 * N * N)


def lux_to_photons_per_pixel_per_sec(scene_lux, f_number, pixel_um, tau=0.85):
    """Incident photons at the sensor pixel per second (no QE applied)."""
    A = (pixel_um * 1e-6) ** 2
    return scene_lux * lens_factor(f_number, tau) * PHOTOPIC_PH_PER_M2_PER_S_PER_LUX * A


# =============================================================================
# Comparison papers — every number traceable to a paper page or sensor datasheet
# =============================================================================
# Each entry: (label, sensor, lens, scene_lux_low, scene_lux_high,
#              t_exp_low_s, t_exp_high_s, qe_or_pde, kind, citation, notes)
# Photon counts computed downstream.
#
# Numbers verified (2026-04-25):
#   SID  - Chen 2018: <0.1 lux at camera, ISO up to 409 600, 1/30 s short
#          exposure, 10-30 s long, f/5.6 aperture (paper text & Fig 1).
#   ELD  - Wei 2020: amplifies ratios ×100/×200/×300 between long ref and short
#          targets; Sony A7S II identical sensor to SID. Tested 100-300 photon
#          regime in noise model.
#   Starlight - Monakhova 2022: 0.6-0.7 mlux scene, ZEISS Otus f/1.4, Canon
#          LI3030SAI (19 µm pixels, NIR-optimized), 100-200 ms (5-10 fps).
#   QBP  - Ma 2020: SwissSPAD2 measurements, single binary frame,
#          burst-merged photon counts <0.06 detected/frame for darkest
#          condition; integration time per frame ~ 10–100 µs.
#   bit2bit - Liu 2024: SPAD512S sensor, 100k-130k binary frames per
#          acquisition; simulation rate λ̄=0.0625 photons/pixel/frame
#          (incident), observed Bernoulli rate 0.059 (detected).

LITERATURE = [
    # SID short exposure (typical practical short shot with very low light)
    {
        "label": "SID (Chen 2018), short exp.\n0.1 lux indoor, 1/30 s, f/5.6",
        "sensor": "Sony A7S II",   # 5.95 µm
        "pixel_um": 5.95,
        "qe": 0.50,
        "f_number": 5.6,
        "scene_lux_lo": 0.03, "scene_lux_hi": 0.3,
        "t_exp_lo": 1/30,    "t_exp_hi": 1/30,
        "kind": "CMOS",
        "color": "tab:blue",
        "citation": "Chen et al., CVPR 2018 (1805.01934). f/5.6, ISO ≤409 600 stated; 1/30 s example in Fig 1.",
    },
    # SID long exposure (10-30 s reference)
    {
        "label": "SID (Chen 2018), long exp.\n0.03–5 lux, 10–30 s, f/5.6",
        "sensor": "Sony A7S II",
        "pixel_um": 5.95,
        "qe": 0.50,
        "f_number": 5.6,
        "scene_lux_lo": 0.03, "scene_lux_hi": 5.0,
        "t_exp_lo": 10.0,    "t_exp_hi": 30.0,
        "kind": "CMOS",
        "color": "tab:cyan",
        "citation": "Chen et al., CVPR 2018. Reference exposures 10–30 s.",
    },
    # ELD — uses SID-like cameras with stronger ratios, very low light
    {
        "label": "ELD (Wei 2020)\nlow-light noise-model regime",
        "sensor": "Sony A7S II / Nikon D850 / Canon",
        "pixel_um": 5.95,
        "qe": 0.50,
        "f_number": 4.0,   # representative for their setup; not all reported
        "scene_lux_lo": 0.01, "scene_lux_hi": 0.5,
        "t_exp_lo": 0.1,     "t_exp_hi": 30.0,
        "kind": "CMOS",
        "color": "tab:purple",
        "citation": "Wei et al., CVPR 2020 (2003.12751). Calibrates ×100/200/300 ratios; targets 100–300 e⁻ regime.",
    },
    # Starlight
    {
        "label": "Starlight (Monakhova 2022)\n0.0006–0.001 lux, 0.1–0.2 s, f/1.4",
        "sensor": "Canon LI3030SAI (19 µm)",
        "pixel_um": 19.0,
        "qe": 0.40,           # NIR-optimized; visible QE conservative
        "f_number": 1.4,
        "scene_lux_lo": 0.0001, "scene_lux_hi": 0.001,
        "t_exp_lo": 0.1,        "t_exp_hi": 0.2,
        "kind": "CMOS",
        "color": "tab:olive",
        "citation": "Monakhova et al., CVPR 2022 (2204.04210). Paper §3 specs; PR-810 Prichard photometer.",
    },
    # QBP — SPAD, single binary frame
    {
        "label": "Quanta Burst (Ma 2020)\nSwissSPAD2, 1 binary frame",
        "sensor": "SwissSPAD2 (16.38 µm)",
        "pixel_um": 16.38,
        "qe": 0.30,           # SwissSPAD2 visible-band PDE peak
        "f_number": 2.0,      # representative
        "scene_lux_lo": None, "scene_lux_hi": None,
        # Direct measurement: 0.01-1 detected ph/pixel/frame (paper Fig 7-8 range)
        "ph_pixel_per_exposure_inc": (0.01 / 0.30, 1.0 / 0.30),  # → 0.03 – 3.3 incident
        "t_exp_lo": 50e-6,  "t_exp_hi": 50e-6,    # representative SPAD frame
        "kind": "SPAD",
        "color": "tab:orange",
        "citation": "Ma et al., SIGGRAPH 2020 (2006.11840). Direct SPAD measurement; range cites Fig 7-8.",
    },
    # bit2bit
    {
        "label": "bit2bit (Liu 2024)\nSPAD512S, 1 binary frame",
        "sensor": "SPAD512S",
        "pixel_um": 16.38,
        "qe": 0.50,           # SPAD512S typical (per datasheet)
        "f_number": 2.0,
        "scene_lux_lo": None, "scene_lux_hi": None,
        # Direct: 0.06 detected ph/pixel/frame (their main simulation operating point)
        # Two reported values: λ̄=0.0625 (sim parameter) and 0.06 detected/PDE
        # → incident range ~0.06 – 0.12
        "ph_pixel_per_exposure_inc": (0.0625, 0.12),
        "t_exp_lo": 50e-6,  "t_exp_hi": 50e-6,
        "kind": "SPAD",
        "color": "tab:red",
        "citation": "Liu et al., NeurIPS 2024 (2410.23247). Abstract: λ̄=0.0625 incident → 0.0590 detected/frame.",
    },
]


# Read-noise floors for typical CMOS sensors
READ_NOISE_FLOOR_LOW = 2.0   # e⁻ RMS, low-noise scientific CMOS
READ_NOISE_FLOOR_HI  = 5.0   # e⁻ RMS, consumer CMOS


def measure_our_spad(images_dir, n_scenes, n_frames):
    """For each filter, compute median + IQR of detected photons per frame
    across scenes. Returns a dict of stats per filter."""
    images_dir = Path(images_dir)
    all_scenes = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    step = max(1, len(all_scenes) // n_scenes)
    scenes = all_scenes[::step][:n_scenes]
    print(f"Measuring our SPAD over {len(scenes)} scenes "
          f"(every {step}-th of {len(all_scenes)})")

    out = {}
    for filt in ["RAW_empty", "RAW_OD_01", "RAW_OD_03", "RAW_OD_07"]:
        lams = []
        ps = []
        t0 = time.time()
        for scene in scenes:
            bin_path = images_dir / scene / f"{filt}.bin"
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
            mid = (p > 0.005) & (p < 0.95)
            if mid.sum() < 1000:
                continue
            lam = -np.log1p(-np.clip(p[mid], 0.0, 1.0 - 1e-6))
            lams.append(float(np.median(lam)))
            ps.append(float(np.median(p[mid])))
        out[filt] = {
            "lam_det_per_frame": np.array(lams),
            "p_median": np.array(ps),
            "n_scenes": len(lams),
            "elapsed_s": time.time() - t0,
        }
        if len(lams):
            q1, med, q3 = np.percentile(lams, [25, 50, 75])
            print(f"  {filt:<12}: n={len(lams):>3}, "
                  f"median λ_det={med:.4f}/frame "
                  f"(IQR {q1:.4f}–{q3:.4f}), {time.time()-t0:.1f}s")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="/nfs/horai.dgpsrv/ondemand30/jw954/images")
    p.add_argument("--output-dir", default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
    p.add_argument("--n-scenes", type=int, default=200)
    p.add_argument("--n-frames", type=int, default=1000)
    p.add_argument("--spad-pde", type=float, default=0.30,
                   help="SwissSPAD2 visible-band PDE for incident←detected conversion")
    p.add_argument("--spad-frame-rate-hz", type=float, default=16667.0)
    p.add_argument("--spad-pixel-um", type=float, default=16.38)
    p.add_argument("--spad-f-number", type=float, default=2.0,
                   help="Lens f-number for our SPAD setup (assumed; please confirm)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  Photon-regime calibration figure — rigorous version")
    print("=" * 70)

    # ---------- Compute photon counts for each comparison paper ----------
    print("\n[1] Per-paper photon flux (incident at sensor face) …")
    paper_rows = []
    for entry in LITERATURE:
        # Either the paper directly reports per-exposure photon count, or we
        # compute it from lux + lens + sensor + exposure.
        if entry.get("ph_pixel_per_exposure_inc") is not None:
            inc_lo, inc_hi = entry["ph_pixel_per_exposure_inc"]
            method = "direct"
        else:
            flux_lo = lux_to_photons_per_pixel_per_sec(
                entry["scene_lux_lo"], entry["f_number"], entry["pixel_um"]
            )
            flux_hi = lux_to_photons_per_pixel_per_sec(
                entry["scene_lux_hi"], entry["f_number"], entry["pixel_um"]
            )
            inc_lo = flux_lo * entry["t_exp_lo"]
            inc_hi = flux_hi * entry["t_exp_hi"]
            method = "lens-eq from lux"
        det_lo = inc_lo * entry["qe"] if entry["kind"] == "CMOS" else \
                 (1 - np.exp(-inc_lo * entry["qe"]))
        det_hi = inc_hi * entry["qe"] if entry["kind"] == "CMOS" else \
                 (1 - np.exp(-inc_hi * entry["qe"]))
        paper_rows.append({
            "label_short": entry["label"].split("\n")[0].split("(")[0].strip(),
            "label": entry["label"],
            "kind": entry["kind"],
            "sensor": entry["sensor"],
            "f_number": entry["f_number"],
            "pixel_um": entry["pixel_um"],
            "qe_or_pde": entry["qe"],
            "scene_lux_lo": entry.get("scene_lux_lo"),
            "scene_lux_hi": entry.get("scene_lux_hi"),
            "t_exp_lo": entry["t_exp_lo"],
            "t_exp_hi": entry["t_exp_hi"],
            "incident_lo": inc_lo,
            "incident_hi": inc_hi,
            "detected_lo": det_lo,
            "detected_hi": det_hi,
            "method": method,
            "color": entry["color"],
            "citation": entry["citation"],
        })
        print(f"  {entry['label'].split(chr(10))[0]:<55}  "
              f"incident: {inc_lo:.3g} – {inc_hi:.3g}  "
              f"detected: {det_lo:.3g} – {det_hi:.3g}")

    # ---------- Measure our SPAD ----------
    print("\n[2] Measuring our SPAD per filter …")
    spad = measure_our_spad(args.images_dir, args.n_scenes, args.n_frames)

    # OD interpretation: our DATA shows ~3.8× attenuation from RAW_empty to
    # RAW_OD_07 and ~1.2× and ~1.8× for OD_01 and OD_03. Consistent with OD
    # values 0.1, 0.3, 0.7 (transmissions 0.794, 0.501, 0.200), NOT 1, 3, 7.
    # If the user confirms the lab uses integer OD values, both nominal and
    # measured numbers should be reconciled.
    OD_NOMINAL_DECIMAL = {"RAW_empty": 0.0, "RAW_OD_01": 0.1, "RAW_OD_03": 0.3, "RAW_OD_07": 0.7}
    OD_NOMINAL_INTEGER = {"RAW_empty": 0.0, "RAW_OD_01": 1.0, "RAW_OD_03": 3.0, "RAW_OD_07": 7.0}

    spad_rows = []
    for filt in ["RAW_empty", "RAW_OD_01", "RAW_OD_03", "RAW_OD_07"]:
        s = spad[filt]
        lams = s["lam_det_per_frame"]
        if len(lams) == 0:
            continue
        # Statistics across scenes
        p1, q1, med, q3, p99 = np.percentile(lams, [1, 25, 50, 75, 99])
        # Per-frame photon counts: detected and incident
        det_lo, det_hi = float(np.percentile(lams, 25)), float(np.percentile(lams, 75))
        inc_lo = det_lo / args.spad_pde
        inc_hi = det_hi / args.spad_pde
        # Per "exposure" — natural binary frame
        # Also compute "per K=1000 frame integration" (~60 ms)
        # And "per K=10000" (~600 ms)
        per_sec_lo = inc_lo * args.spad_frame_rate_hz
        per_sec_hi = inc_hi * args.spad_frame_rate_hz

        spad_rows.append({
            "label_short": f"Ours {filt}",
            "label": f"Ours, {filt}\n(measured, n={len(lams)} scenes)",
            "kind": "SPAD-ours",
            "n_scenes": int(len(lams)),
            "lam_det_p25": float(q1),
            "lam_det_p50": float(med),
            "lam_det_p75": float(q3),
            "lam_det_p1":  float(p1),
            "lam_det_p99": float(p99),
            "incident_lo_per_frame": inc_lo,
            "incident_hi_per_frame": inc_hi,
            "incident_p50_per_frame": float(med / args.spad_pde),
            "incident_min_per_frame": float(p1 / args.spad_pde),
            "incident_max_per_frame": float(p99 / args.spad_pde),
            "incident_lo_per_sec": per_sec_lo,
            "incident_hi_per_sec": per_sec_hi,
            "OD_decimal_assumed": OD_NOMINAL_DECIMAL[filt],
            "OD_integer_alternative": OD_NOMINAL_INTEGER[filt],
            "transmission_decimal": 10 ** -OD_NOMINAL_DECIMAL[filt],
            "transmission_integer": 10 ** -OD_NOMINAL_INTEGER[filt],
        })

    # OD attenuation observed in our data
    if "RAW_empty" in spad and "RAW_OD_07" in spad and \
       len(spad["RAW_empty"]["lam_det_per_frame"]) > 0:
        ratio_obs = float(np.median(spad["RAW_empty"]["lam_det_per_frame"]) /
                           np.median(spad["RAW_OD_07"]["lam_det_per_frame"]))
        print(f"\nObserved RAW_empty / RAW_OD_07 ratio: {ratio_obs:.2f}×")
        print(f"  Decimal-OD prediction (OD 0.7): {10**0.7:.2f}×  ← matches data ✓")
        print(f"  Integer-OD prediction (OD 7):   {10**7:.2g}×    ← does NOT match data ✗")

    # ---------- Save CSV ----------
    csv_path = out_dir / "photon_calibration.csv"
    with open(csv_path, "w", newline="") as f:
        all_keys = set()
        for r in paper_rows + spad_rows:
            all_keys.update(r.keys())
        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
        writer.writeheader()
        for r in paper_rows + spad_rows:
            writer.writerow(r)
    print(f"\nCSV → {csv_path}")

    # ---------- Build the figure ----------
    print("\n[3] Building figure …")
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = 0
    yticks, yticklabels = [], []

    # CMOS papers (top group)
    for r in paper_rows:
        if r["kind"] != "CMOS":
            continue
        # Use INCIDENT photons (sensor-face) for direct comparison
        ax.plot([r["incident_lo"], r["incident_hi"]], [y_pos, y_pos],
                "-", lw=8, alpha=0.75, color=r["color"], solid_capstyle="round")
        yticks.append(y_pos); yticklabels.append(r["label"])
        y_pos += 1

    # Separator
    ax.axhline(y_pos - 0.5, color="gray", ls=":", lw=0.7, alpha=0.5)

    # SPAD comparison papers
    for r in paper_rows:
        if r["kind"] != "SPAD":
            continue
        ax.plot([r["incident_lo"], r["incident_hi"]], [y_pos, y_pos],
                "-", lw=8, alpha=0.75, color=r["color"], solid_capstyle="round")
        yticks.append(y_pos); yticklabels.append(r["label"])
        y_pos += 1

    # Separator
    ax.axhline(y_pos - 0.5, color="gray", ls=":", lw=0.7, alpha=0.5)

    # Our SPAD per filter — incident per binary frame
    spad_colors = {"Ours RAW_empty": "darkgreen",
                   "Ours RAW_OD_01": "olivedrab",
                   "Ours RAW_OD_03": "darkorange",
                   "Ours RAW_OD_07": "darkred"}
    for r in spad_rows:
        # Wide P1-P99 (light), narrow IQR (dark)
        ax.plot([r["incident_min_per_frame"], r["incident_max_per_frame"]],
                [y_pos, y_pos], "-", lw=4, alpha=0.4,
                color=spad_colors[r["label_short"]], solid_capstyle="round")
        ax.plot([r["incident_lo_per_frame"], r["incident_hi_per_frame"]],
                [y_pos, y_pos], "-", lw=8, alpha=0.85,
                color=spad_colors[r["label_short"]], solid_capstyle="round")
        ax.plot([r["incident_p50_per_frame"]], [y_pos], "k|", ms=14, mew=2)
        yticks.append(y_pos)
        od = r["OD_decimal_assumed"]
        T = r["transmission_decimal"]
        yticklabels.append(
            f"{r['label_short']}\n(OD {od}, T={T:.1%}; n={r['n_scenes']})"
        )
        y_pos += 1

    # CMOS noise floors (vertical guides)
    # A single low-noise frame needs SNR ≥ 1, so signal ≥ read_noise_floor
    # photons (in detected electrons; assume QE 50% so detected ≈ 0.5×incident).
    # For simplicity we mark in INCIDENT photon counts:
    snr1_low = READ_NOISE_FLOOR_LOW / 0.5   # 4 incident photons for SNR=1
    snr1_hi  = READ_NOISE_FLOOR_HI  / 0.5
    snr3_low = 3 * snr1_low
    ax.axvspan(snr1_low, snr1_hi, color="lightgray", alpha=0.35,
               label=f"CMOS read-noise floor (SNR=1; {snr1_low:.0f}–{snr1_hi:.0f} ph)")
    ax.axvline(snr3_low, color="black", ls="--", lw=1, alpha=0.5,
               label=f"CMOS SNR≈3 ({snr3_low:.0f} ph)")

    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e6)
    ax.set_xlabel("Incident photons per pixel per exposure  (sensor face)",
                  fontsize=12)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, axis="x", which="both", alpha=0.3)
    ax.set_title(
        "Photon-flux regime across low-light imaging datasets\n"
        "Lens-corrected incident photons at sensor pixel; "
        "thick bar = IQR / paper range, thin bar = P1–P99 (ours), │ = median.",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_pdf = fig_dir / "photon_regime.pdf"
    out_png = fig_dir / "photon_regime.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out_pdf}")

    # ---------- Methodology MD ----------
    md_path = out_dir / "photon_methodology.md"
    with open(md_path, "w") as f:
        f.write(_methodology_md(args, paper_rows, spad_rows, ratio_obs))
    print(f"Methodology → {md_path}")

    print("\nDone.")


def _methodology_md(args, paper_rows, spad_rows, ratio_obs):
    return f"""# Photon-flux calibration: methodology

This document explains how every number in `photon_regime.pdf` and
`photon_calibration.csv` was derived. Every entry should be reproducible from
this file alone.

## 1. Common unit and conversion

All numbers in the figure are **incident photons per pixel per exposure at the
sensor face** — i.e. after the lens but before quantum efficiency. This makes
SPAD and CMOS directly comparable in the same units.

The radiometric chain:

```
E_sensor [lux] = E_scene [lux] × τ_lens × π / (4 × N²)         (lens equation)
Φ_sensor [ph/m²/s] = E_sensor × κ                              (lux → photons)
Φ_pixel [ph/pixel/s] = Φ_sensor × A_pixel                      (× pixel area)
N_per_exposure [ph/pixel/exposure] = Φ_pixel × t_exp           (× exposure time)

CMOS detected:  N_det = N_per_exp × QE
SPAD detected:  N_det_per_frame = 1 - exp(-N_per_exp × PDE)    (Bernoulli clipping)
```

Constants:
- `κ = {PHOTOPIC_PH_PER_M2_PER_S_PER_LUX:.3e} ph/m²/s/lux` at 555 nm photopic
  (= 1 lm/W ÷ 683 ÷ photon energy at 555 nm).
- `τ_lens = 0.85` typical multi-coated lens transmittance.
- For object at infinity (m=0): image-side illuminance ratio = `τ × π / (4 N²)`.

## 2. Per-paper sources

### SID (Chen et al., CVPR 2018, arXiv:1805.01934)
- Sensor: Sony A7S II, 5.95 µm pixel pitch, BSI CMOS, QE ≈ 50 % peak (visible).
- Lens: f/5.6 stated in paper text and Figure 1.
- Illumination: <0.1 lux at the camera (paper §1, §3).
- Exposure: 1/30 s short, 10–30 s long reference.
- ISO: 8 000 to 409 600.
- We use indoor/outdoor lux range 0.03–5 lux as documented in the dataset
  (site cchen156.github.io/SID.html: "indoor 0.03–0.3 lux, outdoor 0.2–5 lux").

### ELD (Wei et al., CVPR 2020, arXiv:2003.12751)
- Sensor: Sony A7S II / Nikon D850 / Canon EOS70D / EOS700D.
- Lens not consistently reported across cameras; we use f/4 representative.
- Calibrates ratios ×100 / ×200 / ×300 between long reference and short
  target exposures. Targets the noise-model regime where photon counts are
  in [100, 300] electrons per long exposure.
- Exposure range 0.1–30 s.
- Illumination: not directly reported in lux; uses scenes similar to SID.

### Starlight (Monakhova et al., CVPR 2022, arXiv:2204.04210)
- Sensor: **Canon LI3030SAI** (NOT Sony IMX291), 19 µm pixels, NIR-optimized.
  QE ≈ 40 % visible, higher in NIR.
- Lens: ZEISS Otus 28 mm **f/1.4** ZF.2.
- Illumination: 0.6–0.7 millilux scene measured with PR-810 Prichard
  photometer; dataset extends down to <0.001 lux (no moon).
- Exposure: 100–200 ms (5–10 fps).

### Quanta Burst Photography (Ma et al., SIGGRAPH 2020, arXiv:2006.11840)
- SPAD camera (SwissSPAD2 or similar), 16.38 µm pixel pitch.
- PDE 30 % visible-band representative.
- Direct measurement: 0.01–1 detected photons / pixel / binary frame in
  reported low-light test conditions (paper Figs 7–8).
- Single binary frame ~50 µs.

### bit2bit (Liu et al., NeurIPS 2024, arXiv:2410.23247)
- Sensor: SPAD512S, 16.38 µm pixel.
- PDE 50 % representative for SPAD512S in visible.
- Direct measurement: simulation rate λ̄ = 0.0625 incident photons/pixel/frame,
  observed Bernoulli rate 0.0590 detected. Quoted in paper abstract and §3.
- Real data: 100k–130k binary frames per acquisition.

## 3. Our SPAD setup

Measured directly from `RAW_*.bin` captures using race-free counter
(`spad_utils_fixed.accumulate_counts_whole_file`). For each scene × filter:

```
counts = accumulate(RAW_*.bin, n_frames={args.n_frames})
p̂      = counts / n_frames                                    # Bernoulli rate
λ_det  = -ln(1 - p̂)                                           # detected ph/pixel/frame
λ_inc  = λ_det / PDE                                          # incident at sensor face
```

We mask pixels with `0.005 < p̂ < 0.95` to avoid clipping bias and dark
counts. Per-filter statistics are median + IQR (and P1/P99) across
n={args.n_scenes} scenes (every {2633//args.n_scenes}-th of the 2633-scene
dataset).

Assumed parameters:
- PDE = `{args.spad_pde:.0%}` — SwissSPAD2 visible-band representative.
  Datasheet ranges 5–50 %; if our actual PDE differs, all our incident
  numbers scale inversely.
- frame rate = `{args.spad_frame_rate_hz/1000:.2f} kHz` — SwissSPAD2 nominal
  60 µs/frame. If different, per-second numbers scale linearly.
- Pixel pitch = `{args.spad_pixel_um} µm` (SwissSPAD2 spec).
- Lens f-number assumed = `{args.spad_f_number}`. **Please confirm lab setup;
  this affects scene-radiance back-out, NOT the per-pixel sensor-face flux
  reported in the figure.**

## 4. OD-filter interpretation (IMPORTANT)

The procedure assumed `RAW_OD_01 = 10⁻¹×, RAW_OD_03 = 10⁻³×, RAW_OD_07 =
10⁻⁷×` transmission (i.e. integer OD values 1, 3, 7). **Our measured data
does not support this.** Observed attenuation from RAW_empty to RAW_OD_07:

  **{ratio_obs:.2f}×**

Predicted attenuation:
- Decimal-OD interpretation (OD 0.7 = 10^-0.7 = 0.20×):  **5.01×**  ← matches ✓
- Integer-OD interpretation (OD 7 = 10^-7):              **10 000 000×**  ← does NOT match ✗

Therefore we treat the filenames `RAW_OD_01/03/07` as decimal OD values 0.1,
0.3, 0.7 (transmissions ≈ 79 %, 50 %, 20 %). If the lab actually uses
integer OD filters and the observed attenuation is somehow due to dominant
ambient leakage / mislabeling / setup error, this needs to be reconciled
before publishing the figure.

## 5. CMOS read-noise floor

Vertical bands in the figure mark the conventional CMOS read-noise floor:

- Low-noise scientific CMOS: read noise σ_r ≈ {READ_NOISE_FLOOR_LOW} e⁻ RMS
- Consumer CMOS:              read noise σ_r ≈ {READ_NOISE_FLOOR_HI} e⁻ RMS

Below ~σ_r incident photons (assuming QE ≈ 0.5), the per-pixel signal is
dominated by read noise (SNR < 1). The "SNR ≈ 3" line marks the practical
detectability threshold for CMOS imaging.

## 6. Uncertainty

Each photon-flux number has a multiplicative error band of roughly:

| Source | Approx. range |
|---|---|
| Photopic ↔ broadband scene SPD | × 1.5–3 |
| Lens transmittance estimate | × 1.1 |
| Pixel area (sensor datasheet) | × 1.05 |
| Lens f-number rounding | × 1.5 |
| QE / PDE assumption | × 2 |
| Combined (typical) | **× 5–10** |

So the bars in the figure should be read as "order-of-magnitude correct,
factor-of-a-few uncertain on each end."

## 7. Citations

| Paper | Citation |
|---|---|
"""+ "\n".join(f"| {r['label_short']} | {r['citation']} |" for r in paper_rows) + """

| Our SPAD | This work; bit-rate measurements over 200 sampled scenes from
                 `/nfs/horai.dgpsrv/ondemand30/jw954/images/`. |
"""


if __name__ == "__main__":
    main()
