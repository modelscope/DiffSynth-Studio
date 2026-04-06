#!/usr/bin/env python3
"""Quick audit: what exactly does the VAE roundtrip do to SPAD binary frames?"""
import torch, numpy as np, sys
from pathlib import Path
from PIL import Image

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

import importlib.util
def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_flux_vae = _import_file('fv', PROJ / 'diffsynth/models/flux_vae.py')
_conv = _import_file('fc', PROJ / 'diffsynth/utils/state_dict_converters/flux_vae.py')
_loader = _import_file('fl', PROJ / 'diffsynth/core/loader/file.py')

device = torch.device('cuda')
dtype = torch.bfloat16
raw_sd = _loader.load_state_dict(str(PROJ / 'models/black-forest-labs/FLUX.1-dev/ae.safetensors'), torch_dtype=dtype, device=str(device))

enc = _flux_vae.FluxVAEEncoder().to(device=device, dtype=dtype).eval()
enc.load_state_dict(_conv.FluxVAEEncoderStateDictConverter(raw_sd), strict=False)
dec = _flux_vae.FluxVAEDecoder().to(device=device, dtype=dtype).eval()
dec.load_state_dict(_conv.FluxVAEDecoderStateDictConverter(raw_sd), strict=False)
del raw_sd

DATASET = Path('/home/jw954/projects/aip-lindell/jw954/spad_dataset')

# Test 3 scenes
test_files = [
    'bits/0724-dgp-001_RAW_empty_frames0-0_p.png',
    'bits/0724-dgp-080_RAW_empty_frames0-0_p.png',
    'bits/0801-bahcor-cor03-27_RAW_empty_frames0-0_p.png',
]
gt_files = [
    'RGB/0724-dgp-001_frames0-19999_linear16.png',
    'RGB/0724-dgp-080_frames0-19999_linear16.png',
    'RGB/0801-bahcor-cor03-27_frames0-19999_linear16.png',
]

for spad_rel, gt_rel in zip(test_files, gt_files):
    spad_path = DATASET / spad_rel
    gt_path = DATASET / gt_rel
    if not spad_path.exists():
        print(f"Skipping {spad_rel} (not found)")
        continue

    print(f"\n{'='*70}")
    print(f"Scene: {spad_rel}")
    print(f"{'='*70}")

    # Load SPAD
    spad_pil = Image.open(spad_path).convert('RGB')
    spad_arr = np.array(spad_pil, dtype=np.float32)  # {0, 255}

    # Load GT
    gt_pil = Image.open(gt_path).convert('RGB')
    gt_arr = np.array(gt_pil, dtype=np.float32)

    # Preprocess (match pipeline exactly)
    def to_tensor(arr):
        t = torch.from_numpy(arr).to(device=device, dtype=dtype)
        t = t / 255.0 * 2.0 - 1.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        return t

    def to_numpy(t):
        out = t.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return ((out + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    spad_t = to_tensor(spad_arr)
    gt_t = to_tensor(gt_arr)

    with torch.no_grad():
        z_spad = enc(spad_t)
        z_gt = enc(gt_t)
        spad_recon = to_numpy(dec(z_spad))
        gt_recon = to_numpy(dec(z_gt))

    # === SPAD Analysis ===
    mask_black = spad_arr[:,:,0] == 0
    mask_white = spad_arr[:,:,0] == 255
    white_frac = mask_white.sum() / mask_black.size

    recon_at_black = spad_recon[:,:,0][mask_black]
    recon_at_white = spad_recon[:,:,0][mask_white]

    print(f"\nSPAD: {mask_black.sum()} black, {mask_white.sum()} white ({white_frac*100:.1f}% white)")
    print(f"  Recon at BLACK: mean={recon_at_black.mean():.1f}, std={recon_at_black.std():.1f}, range=[{recon_at_black.min()},{recon_at_black.max()}]")
    print(f"  Recon at WHITE: mean={recon_at_white.mean():.1f}, std={recon_at_white.std():.1f}, range=[{recon_at_white.min()},{recon_at_white.max()}]")
    print(f"  Recon overall:  unique count={len(np.unique(spad_recon[:,:,0]))}, range=[{spad_recon.min()},{spad_recon.max()}]")

    # PSNR breakdown
    mse_black = np.mean((0.0 - recon_at_black.astype(np.float64))**2)
    mse_white = np.mean((255.0 - recon_at_white.astype(np.float64))**2)
    mse_total = np.mean((spad_arr[:,:,0] - spad_recon[:,:,0].astype(np.float64))**2)
    psnr = 10*np.log10(255**2/mse_total) if mse_total > 0 else float('inf')
    print(f"  MSE: black={mse_black:.1f}, white={mse_white:.1f}, total={mse_total:.1f}")
    print(f"  PSNR: {psnr:.2f} dB")

    # Binary agreement
    recon_binary = (spad_recon[:,:,0] > 128).astype(np.uint8) * 255
    agreement = (recon_binary == spad_arr[:,:,0].astype(np.uint8)).mean()
    print(f"  Binary agreement (>128 threshold): {agreement*100:.2f}%")

    # Spatial correlation
    from scipy.ndimage import uniform_filter
    s1 = uniform_filter(spad_arr[:,:,0], size=8)
    s2 = uniform_filter(spad_recon[:,:,0].astype(float), size=8)
    corr = np.corrcoef(s1.flatten(), s2.flatten())[0,1]
    print(f"  Spatial correlation (8x8 smooth): r={corr:.4f}")

    # === GT Analysis ===
    gt_mse = np.mean((gt_arr - gt_recon.astype(np.float64))**2)
    gt_psnr = 10*np.log10(255**2/gt_mse) if gt_mse > 0 else float('inf')
    print(f"\nGT:  PSNR={gt_psnr:.2f} dB, recon range=[{gt_recon.min()},{gt_recon.max()}]")

    # === Latent analysis ===
    z_s = z_spad.cpu().float().squeeze(0)  # (16,64,64)
    z_g = z_gt.cpu().float().squeeze(0)
    print(f"\nLatent z_spad: mean={z_s.mean():.3f}, std={z_s.std():.3f}, range=[{z_s.min():.3f},{z_s.max():.3f}]")
    print(f"Latent z_gt:   mean={z_g.mean():.3f}, std={z_g.std():.3f}, range=[{z_g.min():.3f},{z_g.max():.3f}]")
    cos = torch.nn.functional.cosine_similarity(z_s.flatten().unsqueeze(0), z_g.flatten().unsqueeze(0))
    l2 = torch.norm(z_s - z_g).item()
    print(f"z_spad vs z_gt: cosine={cos.item():.4f}, L2={l2:.2f}")

    # === KEY INSIGHT: Save pixel-perfect comparison (no matplotlib interpolation) ===
    # Crop a 64x64 patch and save at 1:1 pixel ratio
    r, c = 200, 200
    patch_orig = spad_arr[r:r+64, c:c+64, 0].astype(np.uint8)
    patch_recon = spad_recon[r:r+64, c:c+64, 0]
    patch_diff = np.abs(patch_orig.astype(int) - patch_recon.astype(int)).astype(np.uint8)

    # Scale up 4x for visibility (nearest neighbor)
    from PIL import Image as PILImage
    orig_up = PILImage.fromarray(patch_orig).resize((256, 256), PILImage.NEAREST)
    recon_up = PILImage.fromarray(patch_recon).resize((256, 256), PILImage.NEAREST)
    diff_up = PILImage.fromarray(patch_diff).resize((256, 256), PILImage.NEAREST)

    # Side by side
    combined = PILImage.new('L', (256*3, 256))
    combined.paste(orig_up, (0, 0))
    combined.paste(recon_up, (256, 0))
    combined.paste(diff_up, (512, 0))

    scene_name = Path(spad_rel).stem.split('_')[0]
    combined.save(f'/scratch/jw954/vae_analysis/pixel_audit_{scene_name}.png')
    print(f"\nSaved pixel-perfect patch comparison: pixel_audit_{scene_name}.png")
    print(f"  [Original | Roundtrip | |Difference|] — 64x64 patch at (200,200), 4x zoom NN")

print("\n\n=== CONCLUSION ===")
print("PSNR sensitivity analysis for binary images:")
print("  Shift every pixel by 5:  PSNR=34.1 dB (visually identical)")
print("  Shift every pixel by 1:  PSNR=48.1 dB")
print("  Flip 1% of pixels:       PSNR=20.0 dB (barely visible)")
print("  All grey (128):           PSNR=6.0 dB")
print("")
print("Key: PSNR is VERY sensitive for binary images because any deviation from")
print("{0,255} creates squared error. A pixel at 5 instead of 0 has MSE=25,")
print("while for a natural image the same error (125->130) is invisible.")
print("The VAE smooths sharp binary edges, creating grey halos at boundaries.")
print("This looks identical to humans but tanks PSNR.")
