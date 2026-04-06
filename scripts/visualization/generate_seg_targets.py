#!/usr/bin/env python3
"""Generate segmentation targets for linear probing experiments.

Two target types:
  1. Object presence (global):  binary vectors indicating which objects are in each image
  2. Spatial segmentation:      per-token (32×32) binary masks from SAM3

Usage:
  # Phase 1: object presence only (no GPU needed, uses diffsynth env)
  /home/jw/miniconda3/envs/diffsynth/bin/python generate_seg_targets.py --presence-only \
      --metadata_csv /path/to/metadata_val.csv \
      --dataset_base /path/to/spad_dataset \
      --vlm_objects /path/to/vlm_objects/objects_per_image.jsonl \
      --output-dir ./probing_results_allblocks

  # Phase 2: spatial segmentation masks (needs SAM3, use sam3 env)
  /home/jw/miniconda3/envs/sam3/bin/python generate_seg_targets.py --sam3-masks \
      --metadata_csv /path/to/metadata_val.csv \
      --dataset_base /path/to/spad_dataset \
      --vlm_objects /path/to/vlm_objects/objects_per_image.jsonl \
      --output-dir ./probing_results_allblocks

  # Use full vocabulary for each image (professor's suggestion)
  ... --use-full-vocab
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PATCH_H, PATCH_W = 32, 32   # token grid resolution (matches linear_probing.py)
MIN_FREQ_PCT = 5.0           # minimum frequency (%) to include an object class


def build_vocabulary(vlm_path: Path, val_stems: set, min_freq_pct: float = MIN_FREQ_PCT):
    """Build object vocabulary from VLM annotations, filtered to val set.

    Returns:
        vocab: sorted list of object names meeting frequency threshold
        per_image: dict mapping image stem → list of objects
    """
    freq = Counter()
    per_image = {}
    with open(vlm_path) as f:
        for line in f:
            d = json.loads(line)
            stem = Path(d["image_path"]).stem
            if stem in val_stems:
                per_image[stem] = d["objects"]
                freq.update(d["objects"])

    n = len(per_image)
    threshold = n * min_freq_pct / 100.0

    # Merge labels that differ only by space vs underscore
    norm_map = {}
    for obj, cnt in freq.items():
        norm = obj.replace(" ", "_")
        if norm not in norm_map or cnt > norm_map[norm][1]:
            norm_map[norm] = (obj, cnt)
        else:
            prev_label, prev_cnt = norm_map[norm]
            norm_map[norm] = (prev_label, prev_cnt + cnt)

    label_remap = {}
    for obj in freq:
        norm = obj.replace(" ", "_")
        label_remap[obj] = norm_map[norm][0]

    for stem in per_image:
        per_image[stem] = list({label_remap.get(o, o) for o in per_image[stem]})

    freq_merged = Counter()
    for stem, objs in per_image.items():
        freq_merged.update(objs)

    vocab = sorted([obj for obj, cnt in freq_merged.items() if cnt >= threshold])
    print(f"Vocabulary: {len(vocab)} objects (>= {min_freq_pct}% of {n} val images)")
    for obj in vocab:
        print(f"  {obj}: {freq_merged[obj]} ({freq_merged[obj]/n*100:.1f}%)")
    return vocab, per_image


def add_object_presence(targets: dict, vocab: list, per_image: dict,
                        val_stems: list) -> dict:
    """Add per-class binary presence targets to targets dict."""
    targets["_obj_vocab"] = vocab
    for obj in vocab:
        key = f"obj_{obj.replace(' ', '_')}"
        targets[key] = []
        for stem in val_stems:
            objs = per_image.get(stem, [])
            targets[key].append(1.0 if obj in objs else 0.0)
        n_pos = sum(targets[key])
        print(f"  {key}: {int(n_pos)} positive ({n_pos/len(val_stems)*100:.1f}%)")
    return targets


def add_sam3_masks(targets: dict, vocab: list, per_image: dict,
                   val_stems: list, dataset_base: Path,
                   sam3_checkpoint: str = None, device: str = "cuda",
                   use_full_vocab: bool = False, cache_dir: Path = None):
    """Run SAM3 on GT RGB images to generate spatial segmentation targets.

    Args:
        use_full_vocab: If True, run all vocab prompts on every image (professor's suggestion).
                       If False, only run per-image VLM-detected objects.
        cache_dir: Directory for intermediate .npz cache files (enables resume).
    """
    import torch

    if sam3_checkpoint is None:
        sam3_checkpoint = str(Path.home() / ".cache/huggingface/hub/models--jetjodh--sam3/"
                             "snapshots/1aa50ce07302cb375f85d8084b68a0fb378b8d85/sam3.pt")
    if not Path(sam3_checkpoint).exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {sam3_checkpoint}")

    import sys
    sam3_dir = "/home/jw/engsci/thesis/spad/sam3"
    if sam3_dir not in sys.path:
        sys.path.insert(0, sam3_dir)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"Loading SAM3 model (device={device}) …")
    model = build_sam3_image_model(checkpoint_path=sam3_checkpoint, load_from_HF=False,
                                    device=device)
    processor = Sam3Processor(model, device=device)
    print("SAM3 ready.")

    # Cache directory for resume capability
    if cache_dir is None:
        cache_dir = dataset_base / "RGB" / "sam3_mask_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    n = len(val_stems)
    seg_masks = {obj: np.zeros((n, PATCH_H, PATCH_W), dtype=np.float32) for obj in vocab}
    seg_any = np.zeros((n, PATCH_H, PATCH_W), dtype=np.float32)

    n_cached = 0
    n_processed = 0

    for idx, stem in enumerate(tqdm(val_stems, desc="SAM3 segmentation")):
        cache_file = cache_dir / f"{stem}.npz"

        # Check cache
        if cache_file.exists():
            cached = np.load(cache_file)
            for obj in vocab:
                okey = obj.replace(" ", "_")
                if okey in cached:
                    seg_masks[obj][idx] = cached[okey]
            if "any" in cached:
                seg_any[idx] = cached["any"]
            n_cached += 1
            continue

        # Load GT RGB
        gt_path = dataset_base / "RGB" / f"{stem}.png"
        if not gt_path.exists():
            print(f"  WARNING: GT not found: {gt_path}")
            continue

        image = Image.open(gt_path).convert("RGB")
        state = processor.set_image(image)
        h, w = image.height, image.width

        # Determine which prompts to run
        if use_full_vocab:
            prompts_to_run = vocab
        else:
            img_objects = per_image.get(stem, [])
            prompts_to_run = [obj for obj in img_objects if obj in vocab]

        cache_data = {}
        for obj in prompts_to_run:
            try:
                output = processor.set_text_prompt(state=state, prompt=obj)
                masks = output["masks"]
                if masks is None or masks.numel() == 0:
                    continue
                np_masks = masks.cpu().numpy()
                union = np.zeros((h, w), dtype=bool)
                for m in np_masks:
                    if m.ndim == 4:
                        m = m[0, 0]
                    elif m.ndim == 3:
                        m = m[0]
                    union |= (m > 0.5)
                # Downsample to token grid
                mask_pil = Image.fromarray(union.astype(np.uint8) * 255)
                mask_small = np.array(
                    mask_pil.resize((PATCH_W, PATCH_H), Image.NEAREST),
                    dtype=np.float32,
                ) / 255.0
                seg_masks[obj][idx] = mask_small
                seg_any[idx] = np.maximum(seg_any[idx], mask_small)
                cache_data[obj.replace(" ", "_")] = mask_small
            except Exception as e:
                print(f"  WARNING: SAM3 failed for '{obj}' on {stem}: {e}")

        cache_data["any"] = seg_any[idx]
        np.savez_compressed(cache_file, **cache_data)
        n_processed += 1

    print(f"  Processed: {n_processed}, cached: {n_cached}")

    # Store as targets
    for obj in vocab:
        key = f"spatial_seg_{obj.replace(' ', '_')}"
        targets[key] = seg_masks[obj].tolist()
        coverage = (seg_masks[obj] > 0.5).any(axis=(1, 2)).sum()
        print(f"  {key}: masks in {int(coverage)}/{n} images")

    targets["spatial_seg_any"] = seg_any.tolist()
    coverage_any = (seg_any > 0.5).any(axis=(1, 2)).sum()
    print(f"  spatial_seg_any: masks in {int(coverage_any)}/{n} images")

    return targets


def main():
    parser = argparse.ArgumentParser(description="Generate segmentation targets for probing")
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--dataset_base", required=True)
    parser.add_argument("--vlm_objects", required=True,
                        help="Path to objects_per_image.jsonl")
    parser.add_argument("--output-dir", required=True, nargs="+",
                        help="Probing output dir(s) containing targets.json")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--min-freq-pct", type=float, default=MIN_FREQ_PCT)
    parser.add_argument("--presence-only", action="store_true",
                        help="Only add object presence (global), skip SAM3 masks")
    parser.add_argument("--sam3-masks", action="store_true",
                        help="Generate SAM3 spatial segmentation masks")
    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--device", default="cuda", help="Device for SAM3 (cuda or cpu)")
    parser.add_argument("--use-full-vocab", action="store_true",
                        help="Run all vocab prompts on every image (not just VLM-detected)")
    args = parser.parse_args()

    csv_path = Path(args.metadata_csv)
    dbase = Path(args.dataset_base)
    with open(csv_path) as f:
        samples = list(csv.DictReader(f))
    if args.max_samples:
        samples = samples[:args.max_samples]

    val_stems = [Path(s["image"]).stem for s in samples]
    val_stems_set = set(val_stems)
    print(f"Validation samples: {len(val_stems)}")

    vlm_path = Path(args.vlm_objects)
    vocab, per_image = build_vocabulary(vlm_path, val_stems_set, args.min_freq_pct)

    # Process each output directory
    for out_dir_str in args.output_dir:
        out_dir = Path(out_dir_str)
        tf = out_dir / "targets.json"
        if tf.exists():
            print(f"\nLoading existing targets from {tf}")
            with open(tf) as f:
                targets = json.load(f)
        else:
            print(f"\nNo existing targets.json at {tf}, creating new")
            targets = {}

        print(f"\n=== Adding object presence targets → {out_dir} ===")
        targets = add_object_presence(targets, vocab, per_image, val_stems)

        if args.sam3_masks and not args.presence_only:
            print(f"\n=== Generating SAM3 spatial segmentation masks → {out_dir} ===")
            targets = add_sam3_masks(
                targets, vocab, per_image, val_stems, dbase,
                sam3_checkpoint=args.sam3_checkpoint,
                device=args.device,
                use_full_vocab=args.use_full_vocab,
            )

        with open(tf, "w") as f:
            json.dump(targets, f)
        print(f"Targets saved → {tf}")

    n_global = sum(1 for k in targets if k.startswith("obj_"))
    n_spatial = sum(1 for k in targets if k.startswith("spatial_seg_"))
    print(f"\n  Global object presence targets: {n_global}")
    print(f"  Spatial segmentation targets: {n_spatial}")


if __name__ == "__main__":
    main()
