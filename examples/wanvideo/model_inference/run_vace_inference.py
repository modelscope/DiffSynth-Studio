import argparse
import os
from pathlib import Path

import torch
import imageio
from PIL import Image

from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


def str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def maybe_load_frames(video_path: str | None, height: int, width: int, num_frames: int):
    if not video_path:
        return None
    vd = VideoData(video_path, height=height, width=width)
    return [vd[i] for i in range(num_frames)]


def maybe_load_reference_image(image_path: str | None, height: int, width: int):
    if not image_path:
        return None
    if os.path.isdir(image_path):
        images = []
        for filename in sorted(os.listdir(image_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                file_path = os.path.join(image_path, filename)
                img = Image.open(file_path).convert("RGB")
                images.append(img.resize((width, height)))
        return images
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
    p.add_argument("--model_id", default="Wan-AI/Wan2.1-VACE-14B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument("--vace_in_dim", type=int, required=True, choices=[96, 112, 128])

    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="色调艳丽，过曝，细节模糊不清，字幕，风格，作品，画作，画面，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

    p.add_argument("--vace_video", default=None)
    p.add_argument("--vace_video_mask", default=None)
    p.add_argument("--depth_video", default=None)
    p.add_argument("--normal_video", default=None)
    p.add_argument("--vace_reference_image", default=None)

    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tiled", type=str2bool, default=True)

    p.add_argument("--lora_path", default=None)
    p.add_argument("--lora_alpha", type=float, default=1.0)

    p.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output FPS. Use 0 (default) to auto-detect from --vace_video.",
    )
    p.add_argument("--output", required=True)

    args = p.parse_args()

    os.environ["DIFFSYNTH_WAN_VACE_IN_DIM"] = str(args.vace_in_dim)

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    vace_video = maybe_load_frames(args.vace_video, args.height, args.width, args.num_frames)
    vace_video_mask = maybe_load_frames(args.vace_video_mask, args.height, args.width, args.num_frames)
    depth_video = maybe_load_frames(args.depth_video, args.height, args.width, args.num_frames)
    normal_video = maybe_load_frames(args.normal_video, args.height, args.width, args.num_frames)
    vace_reference_image = maybe_load_reference_image(args.vace_reference_image, args.height, args.width)

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        vace_video=vace_video,
        vace_video_mask=vace_video_mask,
        depth_video=depth_video,
        normal_video=normal_video,
        vace_reference_image=vace_reference_image,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
        tiled=args.tiled,
    )

    out_fps = float(args.fps)
    if out_fps <= 0:
        inferred = get_video_fps(args.vace_video) if args.vace_video else None
        if inferred is None:
            out_fps = 8.0
            print(f"[WARN] Cannot infer FPS from vace_video; fallback to {out_fps}")
        else:
            out_fps = float(inferred)
    print(f"Saving with fps={out_fps}")
    save_video(video, str(out_path), fps=out_fps, quality=5)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
