import argparse
import csv
import os
from pathlib import Path

import torch
import imageio
from PIL import Image
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


def str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def resolve_path(value: str | None, *, video_root: Path, csv_dir: Path) -> Path | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return p

    # Most metadata files use relative paths like "./DL3DV-10K/..."
    # Prefer video_root when provided; otherwise fallback to CSV folder.
    base = video_root if str(video_root) else csv_dir
    return (base / p).resolve()


def maybe_load_frames(video_path: str | None, height: int, width: int, num_frames: int):
    if not video_path:
        return None
    vd = VideoData(video_path, height=height, width=width)
    return [vd[i] for i in range(num_frames)]


def maybe_load_reference_image(image_path: str | None, height: int, width: int):
    """Load one or multiple reference images.
    
    If image_path contains '|', split by '|' to load multiple images as a list.
    Otherwise, load a single image.
    """
    if not image_path:
        return None
    
    # Check if multiple paths (separated by '|')
    if "|" in image_path:
        paths = [p.strip() for p in image_path.split("|")]
        images = []
        for p in paths:
            if p:
                img = Image.open(p).convert("RGB")
                img = img.resize((width, height))
                images.append(img)
        return images if images else None
    else:
        # Single image
        img = Image.open(image_path).convert("RGB")
        return img.resize((width, height))


def get_video_fps(video_path: str) -> float | None:
    """Best-effort FPS probe for an input video.

    Returns None if FPS cannot be determined.
    """
    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data() or {}
        reader.close()
        fps = meta.get("fps", None)
        if fps is None:
            return None
        fps = float(fps)
        if fps <= 1e-6:
            return None
        return fps
    except Exception:
        return None


def maybe_load_lora_and_patch_embedder(pipe: WanVideoPipeline, lora_path: str | None, alpha: float, device: str):
    if not lora_path:
        return

    sd = load_state_dict(lora_path, torch_dtype=pipe.torch_dtype, device=device)
    pipe.load_lora(pipe.vace, state_dict=sd, alpha=alpha)

    # If you finetuned VACE context embedder (vace_patch_embedding), load it too.
    for prefix in ("", "vace."):
        w_key = f"{prefix}vace_patch_embedding.weight"
        b_key = f"{prefix}vace_patch_embedding.bias"
        if w_key in sd:
            patch_sd = {"weight": sd[w_key]}
            if b_key in sd:
                patch_sd["bias"] = sd[b_key]
                print(f"Loading finetuned VACE embedder bias from {lora_path} ({prefix or 'no prefix'})")
            print(f"Loading finetuned VACE embedder weights from {lora_path} ({prefix or 'no prefix'})")
            pipe.vace.vace_patch_embedding.load_state_dict(patch_sd, strict=False)
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv", required=True)
    p.add_argument(
        "--video_root",
        default=None,
        help="Base folder to resolve relative paths in CSV (e.g., /eva_data1/lynn/test_videos)",
    )

    p.add_argument("--model_id", default="Wan-AI/Wan2.1-VACE-14B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--vace_in_dim", type=int, required=True, choices=[96, 112, 128])

    p.add_argument("--negative_prompt", default="色调艳丽，闪烁，忽明忽暗，鬼影，模糊，结构扭曲，变形，突变，过曝，细节模糊不清，风格，作品，画作，画面，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走")
    p.add_argument("--prompt_column", default="prompt")

    # Decide which CSV column to use as VACE input video.
    # - vace_video: use CSV column `vace_video` as VACE input (default)
    # - depth_video: use CSV column `depth_video` as VACE input (depth corresponds to vace_video)
    p.add_argument("--vace_video_source", choices=["vace_video", "depth_video"], default="vace_video")

    p.add_argument("--use_vace_video_mask", type=str2bool, default=True)
    p.add_argument("--use_depth_video", type=str2bool, default=True)
    p.add_argument("--use_blending_area_mask", type=str2bool, default=True)
    p.add_argument("--use_normal_video", type=str2bool, default=False)
    p.add_argument("--use_vace_reference_image", type=str2bool, default=True)

    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tiled", type=str2bool, default=True)

    p.add_argument("--lora_path", default=None)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--finetune_model", default=None, help="Path to full finetuned safetensors to load into pipe.vace")

    p.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output FPS. Use 0 (default) to auto-detect from each row's vace_video.",
    )
    p.add_argument("--output_dir", required=True)

    p.add_argument("--start", type=int, default=0, help="Start row index (0-based, inclusive)")
    p.add_argument("--end", type=int, default=None, help="End row index (0-based, exclusive)")
    p.add_argument("--continue_on_error", type=str2bool, default=True)
    p.add_argument("--dry_run", type=str2bool, default=False)

    args = p.parse_args()

    os.environ["DIFFSYNTH_WAN_VACE_IN_DIM"] = str(args.vace_in_dim)

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    csv_path = Path(args.metadata_csv)
    csv_dir = csv_path.parent
    video_root = Path(args.video_root).resolve() if args.video_root else csv_dir.resolve()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build pipeline once.
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
    )
    maybe_load_lora_and_patch_embedder(pipe, args.lora_path, args.lora_alpha, device=args.device)

    # If a full finetuned safetensors (checkpoint) is provided, load it into pipe.vace.
    if args.finetune_model:
        print(f"Loading full finetuned checkpoint: {args.finetune_model}")
        sd = load_state_dict(args.finetune_model, torch_dtype=pipe.torch_dtype, device=args.device)
        try:
            pipe.vace.load_state_dict(sd)
            print("Loaded finetuned checkpoint into pipe.vace")
        except Exception as e:
            # Fall back to attempting to strip common prefixes (e.g., 'pipe.vace.') if present
            print(f"Warning: direct load failed: {e}. Trying to adjust keys and retry...")
            new_sd = {}
            for k, v in sd.items():
                nk = k
                if k.startswith("pipe.vace."):
                    nk = k[len("pipe.vace."):]
                elif k.startswith("vace."):
                    nk = k[len("vace."):]
                new_sd[nk] = v
            pipe.vace.load_state_dict(new_sd)
            print("Loaded finetuned checkpoint after key adjustments")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    start = max(args.start, 0)
    end = args.end if args.end is not None else len(rows)
    end = min(end, len(rows))
    total_assigned = max(0, end - start)
    rank = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))

    print(f"[RANK {rank}] assigned rows: start={start}, end={end}, total={total_assigned}")

    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            total=total_assigned,
            desc=f"rank {rank}",
            dynamic_ncols=True,
            leave=True,
        )

    try:
        for done_count, index in enumerate(range(start, end), start=1):
            row = rows[index]
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"row={index}")

            try:
                prompt = row.get(args.prompt_column, "")
                if not prompt:
                    raise ValueError(f"Missing prompt column '{args.prompt_column}'")

                # Resolve paths
                vace_video_path = resolve_path(row.get(args.vace_video_source), video_root=video_root, csv_dir=csv_dir)
                if vace_video_path is None:
                    raise ValueError(f"Missing CSV column '{args.vace_video_source}'")

                vace_video_mask_path = (
                    resolve_path(row.get("vace_video_mask"), video_root=video_root, csv_dir=csv_dir)
                    if args.use_vace_video_mask
                    else None
                )
                depth_video_path = (
                    resolve_path(row.get("depth_video"), video_root=video_root, csv_dir=csv_dir) if args.use_depth_video else None
                )
                normal_video_path = (
                    resolve_path(row.get("normal_video"), video_root=video_root, csv_dir=csv_dir)
                    if args.use_normal_video
                    else None
                )
                vace_ref_path = (
                    resolve_path(row.get("vace_reference_image"), video_root=video_root, csv_dir=csv_dir)
                    if args.use_vace_reference_image
                    else None
                )
                blending_area_mask_path = (
                    resolve_path(row.get("blending_area_mask"), video_root=video_root, csv_dir=csv_dir)
                    if args.use_blending_area_mask
                    else None
                )

                # Create an id from the original `video` column when possible.
                # This is typically something like ./DL3DV-10K/<scene_id>/input_video.mp4
                src_for_id = resolve_path(row.get("video"), video_root=video_root, csv_dir=csv_dir) or vace_video_path
                inferred_id = src_for_id.parent.name if src_for_id.parent.name else src_for_id.stem
                out_name = f"{inferred_id}.mp4"
                out_path = (out_dir / out_name).resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)

                progress_pct = (done_count / total_assigned * 100.0) if total_assigned > 0 else 100.0

                print("=" * 80)
                print(f"[RANK {rank}] progress={done_count}/{total_assigned} ({progress_pct:.1f}%)")
                print(f"[{index}] id={inferred_id}")
                print(f"  vace_video_source={args.vace_video_source}")
                print(f"  vace_video={vace_video_path}")
                if vace_video_mask_path:
                    print(f"  vace_video_mask={vace_video_mask_path}")
                if depth_video_path:
                    print(f"  depth_video={depth_video_path}")
                if normal_video_path:
                    print(f"  normal_video={normal_video_path}")
                if vace_ref_path:
                    print(f"  vace_reference_image={vace_ref_path}")
                if blending_area_mask_path:
                    print(f"  blending_area_mask={blending_area_mask_path}")
                print(f"  output={out_path}")

                out_fps = float(args.fps)
                if out_fps <= 0:
                    inferred = get_video_fps(str(vace_video_path))
                    if inferred is None:
                        out_fps = 8.0
                        print(f"  [WARN] Cannot infer FPS from vace_video; fallback to {out_fps}")
                    else:
                        out_fps = float(inferred)
                print(f"  fps={out_fps}")

                if args.dry_run:
                    continue

                # Load frames
                vace_video = maybe_load_frames(str(vace_video_path), args.height, args.width, args.num_frames)
                vace_video_mask = (
                    maybe_load_frames(str(vace_video_mask_path), args.height, args.width, args.num_frames)
                    if vace_video_mask_path
                    else None
                )
                depth_video = (
                    maybe_load_frames(str(depth_video_path), args.height, args.width, args.num_frames) if depth_video_path else None
                )
                normal_video = (
                    maybe_load_frames(str(normal_video_path), args.height, args.width, args.num_frames) if normal_video_path else None
                )
                vace_reference_image = (
                    maybe_load_reference_image(str(vace_ref_path), args.height, args.width) if vace_ref_path else None
                )
                blending_area_mask = (
                    maybe_load_frames(str(blending_area_mask_path), args.height, args.width, args.num_frames)
                    if blending_area_mask_path
                    else None
                )

                video = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    vace_video=vace_video,
                    vace_video_mask=vace_video_mask,
                    depth_video=depth_video,
                    blending_area_mask=blending_area_mask,
                    normal_video=normal_video,
                    vace_reference_image=vace_reference_image,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    seed=args.seed,
                    tiled=args.tiled,
                )

                save_video(video, str(out_path), fps=out_fps, quality=5)
                print(f"[RANK {rank}] Saved: {out_path}")

            except Exception as e:
                progress_pct = (done_count / total_assigned * 100.0) if total_assigned > 0 else 100.0
                print(f"[RANK {rank}] [ERROR] progress={done_count}/{total_assigned} ({progress_pct:.1f}%) Row {index} failed: {e}")
                if not args.continue_on_error:
                    raise
    finally:
        if progress_bar is not None:
            progress_bar.close()


if __name__ == "__main__":
    main()
