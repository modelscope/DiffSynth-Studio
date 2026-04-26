# Photon-flux calibration: methodology

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
- `κ = 4.087e+15 ph/m²/s/lux` at 555 nm photopic
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
counts = accumulate(RAW_*.bin, n_frames=1000)
p̂      = counts / n_frames                                    # Bernoulli rate
λ_det  = -ln(1 - p̂)                                           # detected ph/pixel/frame
λ_inc  = λ_det / PDE                                          # incident at sensor face
```

We mask pixels with `0.005 < p̂ < 0.95` to avoid clipping bias and dark
counts. Per-filter statistics are median + IQR (and P1/P99) across
n=50 scenes (every 52-th of the 2633-scene
dataset).

Assumed parameters:
- PDE = `30%` — SwissSPAD2 visible-band representative.
  Datasheet ranges 5–50 %; if our actual PDE differs, all our incident
  numbers scale inversely.
- frame rate = `16.67 kHz` — SwissSPAD2 nominal
  60 µs/frame. If different, per-second numbers scale linearly.
- Pixel pitch = `16.38 µm` (SwissSPAD2 spec).
- Lens f-number assumed = `2.0`. **Please confirm lab setup;
  this affects scene-radiance back-out, NOT the per-pixel sensor-face flux
  reported in the figure.**

## 4. OD-filter interpretation (IMPORTANT)

The procedure assumed `RAW_OD_01 = 10⁻¹×, RAW_OD_03 = 10⁻³×, RAW_OD_07 =
10⁻⁷×` transmission (i.e. integer OD values 1, 3, 7). **Our measured data
does not support this.** Observed attenuation from RAW_empty to RAW_OD_07:

  **3.94×**

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

- Low-noise scientific CMOS: read noise σ_r ≈ 2.0 e⁻ RMS
- Consumer CMOS:              read noise σ_r ≈ 5.0 e⁻ RMS

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
| SID | Chen et al., CVPR 2018 (1805.01934). f/5.6, ISO ≤409 600 stated; 1/30 s example in Fig 1. |
| SID | Chen et al., CVPR 2018. Reference exposures 10–30 s. |
| ELD | Wei et al., CVPR 2020 (2003.12751). Calibrates ×100/200/300 ratios; targets 100–300 e⁻ regime. |
| Starlight | Monakhova et al., CVPR 2022 (2204.04210). Paper §3 specs; PR-810 Prichard photometer. |
| Quanta Burst | Ma et al., SIGGRAPH 2020 (2006.11840). Direct SPAD measurement; range cites Fig 7-8. |
| bit2bit | Liu et al., NeurIPS 2024 (2410.23247). Abstract: λ̄=0.0625 incident → 0.0590 detected/frame. |

| Our SPAD | This work; bit-rate measurements over 200 sampled scenes from
                 `/nfs/horai.dgpsrv/ondemand30/jw954/images/`. |
