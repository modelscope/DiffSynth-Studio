#!/usr/bin/env python3
"""
sRGB → simulated SPAD binary captures
=====================================
Version: v1

End-to-end demonstration of the simulator pipeline:

  online sRGB image
      ↓ inverse sRGB OETF (gamma decode)
  linear RGB
      ↓ apply RGB-to-mono weights (default: v4)
  predicted mono intensity
      ↓ exposure scaling α
  predicted Bernoulli rate p_true
      ↓ (optional) per-pixel noise injection via the LUT (sample p_hat from cell)
  per-pixel rate map p_simulated(x, y)
      ↓ Bernoulli sampling × N frames
  binary frame stack
      ↓ pack to MSB-first 1-bit format
  scene_id/RAW_empty.bin

The output .bin files match the format of our real SPAD captures
(512×512 pixels, 1 bit/pixel, MSB-first, 32 768 bytes per frame), so the
existing extract_binary_images.py can be run on them directly to produce
PNG visualizations at various frame-accumulation levels.

Inputs:  AFHQ images (any 512×512 sRGB JPG/PNG)
Weights: defaults to rgb_to_mono_weights_v4.npz (clean, available now);
         v3 weights are physically preferred and will be used once available.
Noise:   defaults to no LUT injection (use p_true directly);
         pass --use-lut to inject per-pixel sampled p_hat.
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# Provenance helper (for loading LUTs)
sys.path.insert(0, "/nfs/horai.dgpsrv/ondemand30/jw954/calibration")

H, W = 512, 512
BYTES_PER_FRAME = (H * W) // 8       # 32 768
N_BINS = 256
N_SUPER = 64
SUPER_PX = 8
VERSION = "v1"


# ---------------------------------------------------------------------------
# Color-space helpers
# ---------------------------------------------------------------------------
def srgb_to_linear(srgb):
    """Inverse sRGB OETF. srgb in [0,1] (float32)."""
    a = 0.055
    threshold = 0.04045
    out = np.where(
        srgb <= threshold,
        srgb / 12.92,
        ((srgb + a) / (1.0 + a)) ** 2.4,
    )
    return out.astype(np.float32)


def linear_to_srgb(linear):
    """Forward sRGB OETF for display."""
    a = 0.055
    out = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        (1.0 + a) * np.clip(linear, 0, None) ** (1.0 / 2.4) - a,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Bit packing (MSB-first, matching our SPAD format)
# ---------------------------------------------------------------------------
def pack_frames_to_bytes(frames_uint8):
    """
    Pack a (N, H, W) uint8 array of 0/1 values into a contiguous packed-bit
    byte stream of N * (H*W // 8) bytes, MSB-first within each byte.
    """
    n, h, w = frames_uint8.shape
    flat = frames_uint8.reshape(n, h * w).astype(np.uint8)
    packed = np.packbits(flat, axis=1, bitorder="big")  # (N, H*W//8)
    return packed.tobytes()


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------
def srgb_to_p_true(img_srgb_uint8, weights, alpha):
    """Map an sRGB uint8 image to a per-pixel Bernoulli rate p_true.

    Pipeline:  sRGB → linear RGB → mono via weights → p = 1 - exp(-alpha * mono)
    """
    srgb = img_srgb_uint8.astype(np.float32) / 255.0
    lin = srgb_to_linear(srgb)
    # weights = (w_r, w_g, w_b)
    mono_lin = (lin[..., 0] * weights[0]
                + lin[..., 1] * weights[1]
                + lin[..., 2] * weights[2])
    mono_lin = np.clip(mono_lin, 0.0, None)
    p_true = 1.0 - np.exp(-alpha * mono_lin)
    return p_true.astype(np.float32), mono_lin.astype(np.float32)


def maybe_inject_lut_noise(p_true, lut_data, lut_kind="super_pixel"):
    """For each pixel, sample one p_hat from the LUT cell that matches its
    p_true bin. Returns the per-pixel sampled rate map.

    lut_data: dict with flat_values, offsets, shape, kind.
    """
    flat = lut_data["flat_values"]
    offs = lut_data["offsets"]

    bin_idx = np.clip((p_true * 255.0 + 0.5).astype(np.int32), 0, N_BINS - 1)

    if lut_kind == "super_pixel":
        # cell = sy * N_SUPER * N_BINS + sx * N_BINS + bin
        sy = (np.arange(H) // SUPER_PX)[:, None]
        sx = (np.arange(W) // SUPER_PX)[None, :]
        cell = sy * N_SUPER * N_BINS + sx * N_BINS + bin_idx
    elif lut_kind == "global":
        cell = bin_idx
    elif lut_kind == "per_pixel":
        yy = np.arange(H)[:, None]
        xx = np.arange(W)[None, :]
        cell = yy * W * N_BINS + xx * N_BINS + bin_idx
    else:
        raise ValueError(lut_kind)

    cell = cell.ravel()
    p_simulated = np.empty_like(p_true).ravel()

    # Vectorised lookup with random sample within each cell
    rng = np.random.default_rng(42)
    starts = offs[cell]
    ends = offs[cell + 1]
    counts = ends - starts
    # For empty cells fall back to p_true
    empty = counts == 0
    nonempty = ~empty
    # Random offsets within each non-empty cell
    rand_offs = rng.integers(0, np.maximum(counts, 1)).astype(np.int64)
    rand_offs[empty] = 0
    sample_idx = starts + rand_offs
    p_simulated[nonempty] = flat[sample_idx[nonempty]]
    p_simulated[empty] = p_true.ravel()[empty]
    return p_simulated.reshape(H, W).astype(np.float32)


def simulate_binary_frames(p_per_pixel, n_frames, seed=0, inverse_rotate_k=0):
    """Sample n_frames i.i.d. Bernoulli frames given per-pixel rate map.

    `inverse_rotate_k`: rotate each frame by np.rot90(k=inverse_rotate_k)
    BEFORE returning. Use this to undo the rotation that extract_binary_images.py
    will apply when it reads the output .bin (rot90(k=1)). Setting
    inverse_rotate_k = -1 (or +3) here will make the round-trip identity.

    Returns (n_frames, H, W) uint8 array of 0/1.
    """
    rng = np.random.default_rng(seed)
    p_clip = np.clip(p_per_pixel, 0.0, 1.0)
    # Generate in chunks to avoid materialising huge intermediates
    CHUNK = 200
    out = np.empty((n_frames, H, W), dtype=np.uint8)
    for s in range(0, n_frames, CHUNK):
        e = min(s + CHUNK, n_frames)
        rs = rng.random(size=(e - s, H, W), dtype=np.float32)
        chunk = (rs < p_clip).astype(np.uint8)
        if inverse_rotate_k % 4 != 0:
            chunk = np.rot90(chunk, k=inverse_rotate_k, axes=(1, 2))
        out[s:e] = chunk
    return out


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--afhq-dir", default="/scratch/ondemand30/jw954/afhq",
                   help="AFHQ root containing train/val/{cat,dog,wild}")
    p.add_argument("--output-dir", default="/scratch/ondemand30/jw954/afhq_simulation",
                   help="Where to write binaries, extracts, and figures")
    p.add_argument("--weights", default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration/rgb_to_mono_weights_v4.npz",
                   help="Path to rgb_to_mono_weights_v?.npz; defaults to v4")
    p.add_argument("--exposure-alpha", type=float, default=4.0,
                   help="Exposure scaling: p = 1 - exp(-alpha * mono_linear)")
    p.add_argument("--n-frames", type=int, default=10_000,
                   help="Number of binary frames to simulate per scene")
    p.add_argument("--n-images", type=int, default=10,
                   help="Number of AFHQ images to simulate")
    p.add_argument("--use-lut", default=None,
                   help="Path to a single-level LUT NPZ to sample per-pixel p_hat "
                        "noise from (super_pixel or global). Single lookup with "
                        "binary fallback to p_true on empty cells. Mutually "
                        "exclusive with --use-pyramid.")
    p.add_argument("--lut-kind", choices=["super_pixel", "global", "per_pixel"],
                   default="super_pixel",
                   help="Which single-level LUT structure to use for sampling "
                        "(only with --use-lut).")
    p.add_argument("--use-pyramid", default=None,
                   help="Path to a multi-level dyadic LUT pyramid NPZ "
                        "(lut_pyramid.npz). Walks the cascade L=1 → L=2 → … → "
                        "L=512, falling back at the minimum required pool size. "
                        "Mutually exclusive with --use-lut.")
    p.add_argument("--pyramid-min-samples", type=int, default=20,
                   help="Per-cell sample threshold for accepting a pyramid level "
                        "(only with --use-pyramid).")
    p.add_argument("--rotate-k", type=int, default=0,
                   help="Apply np.rot90(k) when computing p_true. The LUT's "
                        "super-pixel coords are in image space, which matches "
                        "untreated AFHQ images, so default is 0 (no rotation).")
    p.add_argument("--inverse-rotate-k", type=int, default=-1,
                   help="Rotation applied to each frame BEFORE packing to .bin. "
                        "Default -1 cancels extract_binary_images.py's default "
                        "rot90(k=1), so the final PNG matches AFHQ orientation.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cherry_pick_images(afhq_dir, n=10, seed=42):
    """Pick a diverse set: ~3 cats, ~3 dogs, ~4 wild, all from val/."""
    rng = np.random.default_rng(seed)
    val = Path(afhq_dir) / "val"
    picks = []
    counts = {"cat": 3, "dog": 3, "wild": 4}
    if n != 10:
        # Distribute roughly equally
        per = n // 3
        rem = n - 3 * per
        counts = {"cat": per + (1 if rem > 0 else 0),
                  "dog": per + (1 if rem > 1 else 0),
                  "wild": per}
    for cls, k in counts.items():
        files = sorted((val / cls).glob("*.jpg"))
        if len(files) >= k:
            idx = rng.choice(len(files), k, replace=False)
            for i in idx:
                picks.append((cls, files[i]))
    return picks[:n]


def main():
    args = parse_args()
    out = Path(args.output_dir)
    bin_dir = out / "binaries"
    sel_dir = out / "selected_images"
    bin_dir.mkdir(parents=True, exist_ok=True)
    sel_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  AFHQ → simulated SPAD  [{VERSION}]")
    print("=" * 70)
    print(f"  AFHQ root:   {args.afhq_dir}")
    print(f"  Output:      {out}")
    print(f"  Weights:     {args.weights}")
    print(f"  α (exposure): {args.exposure_alpha}")
    print(f"  Frames/scene: {args.n_frames}")
    print(f"  N images:    {args.n_images}")
    if args.use_pyramid:
        noise_mode = f"PYRAMID cascade (min_samples={args.pyramid_min_samples})"
    elif args.use_lut:
        noise_mode = f"single-level LUT ({args.lut_kind})"
    else:
        noise_mode = "no (p_true directly)"
    print(f"  Noise mode:  {noise_mode}")
    print()

    if args.use_lut and args.use_pyramid:
        sys.exit("ERROR: --use-lut and --use-pyramid are mutually exclusive")

    # Load weights
    w_npz = np.load(args.weights)
    weights = (float(w_npz["w_r"]), float(w_npz["w_g"]), float(w_npz["w_b"]))
    print(f"Weights: w_r={weights[0]:.4f}, w_g={weights[1]:.4f}, w_b={weights[2]:.4f}, "
          f"sum={sum(weights):.4f}")

    # Noise injection: either single-level LUT, multi-level pyramid, or none
    lut_data = None
    pyramid_sampler = None
    if args.use_lut:
        lut_npz = np.load(args.use_lut)
        lut_data = {
            "flat_values": lut_npz["flat_values"],
            "offsets": lut_npz["offsets"],
        }
        print(f"Loaded single-level LUT: {len(lut_data['flat_values']):,} samples, "
              f"{len(lut_data['offsets'])-1:,} cells")
    elif args.use_pyramid:
        from pyramid_sampler import PyramidSampler
        print(f"Loading pyramid (this can take a few minutes from NFS) …")
        pyramid_sampler = PyramidSampler(
            args.use_pyramid, min_samples=args.pyramid_min_samples, seed=args.seed,
        )

    # Pick images
    picks = cherry_pick_images(args.afhq_dir, n=args.n_images, seed=args.seed)
    print(f"\nSelected {len(picks)} images:")
    for i, (cls, fp) in enumerate(picks):
        print(f"  {i+1:>2}. [{cls:>4}] {fp.name}")

    # Process each
    summary_rows = []
    for i, (cls, fp) in enumerate(picks):
        scene_id = f"afhq_{i+1:02d}_{cls}_{fp.stem}"
        print(f"\n[{i+1}/{len(picks)}] {scene_id}")

        # Load image
        img = Image.open(fp).convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H), Image.BICUBIC)
        srgb = np.array(img)
        # Save a copy for reference
        Image.fromarray(srgb).save(sel_dir / f"{scene_id}.png")

        # Forward model: sRGB → p_true
        p_true, mono_lin = srgb_to_p_true(srgb, weights, args.exposure_alpha)

        # Apply rotation so the LUT super-pixel coords match
        if args.rotate_k % 4 != 0:
            p_true = np.rot90(p_true, k=args.rotate_k).copy()
            mono_lin = np.rot90(mono_lin, k=args.rotate_k).copy()

        # Inject noise: pyramid (full cascade), single-level LUT, or none
        per_level_counts = None
        if pyramid_sampler is not None:
            yy_idx, xx_idx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            ys_flat = yy_idx.ravel()
            xs_flat = xx_idx.ravel()
            bins_flat = np.clip(
                (p_true.ravel() * 255.0 + 0.5).astype(np.int32), 0, N_BINS - 1
            )
            # Reset the sampler RNG per scene so each scene is independent
            pyramid_sampler.rng = np.random.default_rng(args.seed + i * 1000 + 1)
            p_hats_flat, levels_used_flat = pyramid_sampler.sample_batch(
                xs_flat, ys_flat, bins_flat,
                min_samples=args.pyramid_min_samples,
            )
            p_per_pixel = p_hats_flat.reshape(H, W).astype(np.float32)
            per_level_counts = {
                int(L): int((levels_used_flat == L).sum()) for L in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            }
        elif lut_data is not None:
            p_per_pixel = maybe_inject_lut_noise(p_true, lut_data, args.lut_kind)
        else:
            p_per_pixel = p_true

        print(f"  p_true range: [{p_true.min():.4f}, {p_true.max():.4f}], "
              f"mean={p_true.mean():.4f}")
        if pyramid_sampler is not None:
            print(f"  p_simulated range: [{p_per_pixel.min():.4f}, {p_per_pixel.max():.4f}], "
                  f"mean={p_per_pixel.mean():.4f}")
            top = sorted(per_level_counts.items(), key=lambda kv: -kv[1])[:5]
            top_str = "  ".join(f"L={L}:{c}" for L, c in top if c > 0)
            print(f"  Pyramid levels used (top): {top_str}")
        elif lut_data is not None:
            print(f"  p_simulated range: [{p_per_pixel.min():.4f}, {p_per_pixel.max():.4f}], "
                  f"mean={p_per_pixel.mean():.4f}")

        # Sample binary frames; apply inverse rotation so extract restores orientation
        t0 = time.time()
        frames = simulate_binary_frames(
            p_per_pixel, args.n_frames,
            seed=args.seed + i * 1000,
            inverse_rotate_k=args.inverse_rotate_k,
        )
        sim_time = time.time() - t0
        print(f"  Simulated {args.n_frames} frames in {sim_time:.1f}s "
              f"(observed mean = {frames.mean():.4f})")

        # Pack and save in our SPAD format
        scene_dir = bin_dir / scene_id
        scene_dir.mkdir(exist_ok=True)
        bin_path = scene_dir / "RAW_empty.bin"
        packed = pack_frames_to_bytes(frames)
        with open(bin_path, "wb") as f:
            f.write(packed)
        print(f"  Wrote {bin_path}  ({len(packed)/1e6:.1f} MB)")

        row = {
            "i": i + 1, "class": cls, "name": fp.name,
            "scene_id": scene_id,
            "p_true_mean": float(p_true.mean()),
            "p_true_min": float(p_true.min()),
            "p_true_max": float(p_true.max()),
            "obs_rate": float(frames.mean()),
            "n_frames": args.n_frames,
        }
        if per_level_counts is not None:
            for L in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                row[f"pyramid_L{L}"] = per_level_counts.get(L, 0)
        summary_rows.append(row)

    # Save summary CSV
    import csv
    csv_path = out / "simulation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved → {csv_path}")
    print(f"\nTotal binaries written: {len(picks)} scenes × {args.n_frames} frames")
    print(f"  Each scene_dir/{...}/RAW_empty.bin can be processed by")
    print(f"  extract_binary_images.py as if it were a real capture.")


if __name__ == "__main__":
    main()
