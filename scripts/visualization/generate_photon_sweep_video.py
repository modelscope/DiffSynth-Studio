#!/usr/bin/env python3
"""Photon-sweep video: OD0.7 (darkest) → 1000 frames (brightest).

Two scenes shown one at a time (full-width), cycling 2-3 per photon level.
0.5s per image. Forward then reverse (boomerang).

Levels: OD0.7 → OD0.3 → OD0.1 → 1 frame → 4 → 16 → 64 → 256 → 1000
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
import subprocess

# -- Config -----------------------------------------------------------------
REPO = Path("/home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD")
OUT_DIR = REPO / "agent"

import sys

FPS = 24
HOLD_SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
SCENE_ID = sys.argv[2] if len(sys.argv) > 2 else "650"  # "650" or "385"
HOLD_FRAMES = round(HOLD_SEC * FPS)
FADE_FRAMES = 4      # quick ~0.17s crossfade between levels

CANVAS_W = 1920
CANVAS_H = 1080

# Colors
BG = (255, 255, 255)
ACCENT = (26, 82, 118)
LABEL_COL = (90, 90, 90)
BORDER = (180, 180, 180)
METER_BG = (230, 230, 230)
METER_FILL = (41, 128, 185)
TICK_COL = (130, 130, 130)

# -- Scene ------------------------------------------------------------------
SCENE_MAP = {
    "650": {"val_idx": 650, "od_idx": 1, "od_dir": "validation_outputs_od_video",
            "name": "Outdoor (Food Truck)"},
    "385": {"val_idx": 385, "od_idx": 0, "od_dir": "validation_outputs_od_video_385",
            "name": "Indoor (Desk)"},
}
SCENES = [SCENE_MAP[SCENE_ID]]

# -- Levels: (source_type, source_dir, label, nframes_equiv) ---------------
#    nframes_equiv is for the photon meter (log scale); OD filters mapped to fractional frames
LEVELS = [
    ("od",    "bits_RAW_OD_07", "OD 0.7 Filter",        0.2),
    ("od",    "bits_RAW_OD_03", "OD 0.3 Filter",        0.5),
    ("od",    "bits_RAW_OD_01", "OD 0.1 Filter",        0.8),
    ("frame", "bits",           "1 Frame (RAW)",         1),
    ("frame", "bits_multi_4",   "4 Frames",              4),
    ("frame", "bits_multi_16",  "16 Frames",             16),
    ("frame", "bits_multi_64",  "64 Frames",             64),
    ("frame", "bits_multi_256", "256 Frames",            256),
    ("frame", "bits_multi_1000","1000 Frames",           1000),
]

# -- Paths ------------------------------------------------------------------
ABLATION = REPO / "validation_outputs_frame_ablation"
BASELINE = REPO / "validation_outputs_scene_aware" / "seed_42"
OD_BASE  = REPO / SCENES[0].get("od_dir", "validation_outputs_od_video")


def load_img(p):
    return np.array(Image.open(p).convert("RGB"))


def get_triplet(level, scene):
    """Return (input, output, gt) paths for a (level, scene) combo."""
    src_type, src_dir, _, _ = level
    if src_type == "od":
        base = OD_BASE / src_dir
        idx = scene["od_idx"]
    elif src_dir == "bits":
        base = BASELINE
        idx = scene["val_idx"]
    else:
        base = ABLATION / src_dir
        idx = scene["val_idx"]
    return (
        base / "input"  / f"input_{idx:04d}.png",
        base / "output" / f"output_{idx:04d}.png",
        base / "ground_truth" / f"gt_{idx:04d}.png",
    )


# -- Font loading -----------------------------------------------------------
def font(size, bold=True):
    tag = "Bold" if bold else "Regular"
    for fam in ["LiberationSans", "DejaVuSans", "FreeSans"]:
        p = f"/usr/share/fonts/truetype/{fam.lower().replace('sans','')}/{fam.replace('Sans','Sans-' if 'Liberation' in fam or 'DejaVu' in fam else '')}{tag}.ttf"
    # Just try common paths
    paths = ([f"/usr/share/fonts/truetype/liberation/LiberationSans-{tag}.ttf",
              f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              f"/usr/share/fonts/truetype/freefont/FreeSans{'Bold' if bold else ''}.ttf"])
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# -- Load all images --------------------------------------------------------
print("Loading images...")
# images[level_idx][scene_idx] = (inp, out, gt) as numpy arrays
all_imgs = []
for li, level in enumerate(LEVELS):
    level_data = []
    for si, scene in enumerate(SCENES):
        ip, op, gp = get_triplet(level, scene)
        level_data.append((load_img(ip), load_img(op), load_img(gp)))
    all_imgs.append(level_data)
    print(f"  {level[2]}")

# -- Layout -----------------------------------------------------------------
# Single scene at a time: 3 images side by side, large
HEADER_H = 56
COL_LABEL_H = 32
METER_H = 64
SIDE_PAD = 40
GAP = 12
TOP_PAD = 8

content_top = HEADER_H + COL_LABEL_H + TOP_PAD
content_bot = CANVAS_H - METER_H
avail_h = content_bot - content_top
avail_w = CANVAS_W - 2 * SIDE_PAD - 2 * GAP

img_sz = min(avail_h, avail_w // 3)
total_w = img_sz * 3 + GAP * 2
x0 = (CANVAS_W - total_w) // 2
y0 = content_top + (avail_h - img_sz) // 2

# Meter
meter_x0 = SIDE_PAD + 100
meter_x1 = CANVAS_W - SIDE_PAD
meter_y = CANVAS_H - METER_H + 12
meter_h = 22

print(f"  Image size: {img_sz}x{img_sz}")

# -- Resize -----------------------------------------------------------------
print("Resizing...")


def rsz(img):
    return np.array(Image.fromarray(img).resize((img_sz, img_sz), Image.LANCZOS))


resized = []  # resized[level][scene] = (inp, out, gt)
for li in range(len(LEVELS)):
    lev = []
    for si in range(len(SCENES)):
        inp, out, gt = all_imgs[li][si]
        lev.append((rsz(inp), rsz(out), rsz(gt)))
    resized.append(lev)
del all_imgs

# -- Fonts ------------------------------------------------------------------
f_title = font(36, bold=True)
f_col = font(22, bold=True)
f_scene = font(18, bold=False)
f_meter_label = font(17, bold=True)
f_meter_tick = font(14, bold=False)

# -- Photon meter -----------------------------------------------------------
# Map all levels to a 0-1 scale. OD filters < 1 frame, multi-frame > 1.
# Use log scale anchored: 0.2 (OD0.7) → 1000
LOG_MIN = math.log10(0.2)
LOG_MAX = math.log10(1000)
LOG_RANGE = LOG_MAX - LOG_MIN


def photon_frac(nf_equiv):
    return (math.log10(max(nf_equiv, 0.1)) - LOG_MIN) / LOG_RANGE


def lerp(a, b, t):
    return a + (b - a) * t


# -- Render -----------------------------------------------------------------
def render(imgs_a, imgs_b, alpha, label, nf_a, nf_b, scene_name):
    """Render one video frame showing a single scene (3 columns)."""
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas)

    # Title
    bbox = draw.textbbox((0, 0), label, font=f_title)
    tw = bbox[2] - bbox[0]
    draw.text(((CANVAS_W - tw) // 2, 10), label, fill=ACCENT, font=f_title)

    # Scene name (small, right-aligned)
    bbox2 = draw.textbbox((0, 0), scene_name, font=f_scene)
    draw.text((CANVAS_W - SIDE_PAD - (bbox2[2] - bbox2[0]), 18),
              scene_name, fill=LABEL_COL, font=f_scene)

    # Column labels
    col_labels = ["SPAD Input", "Reconstruction", "Ground Truth"]
    for c in range(3):
        cx = x0 + c * (img_sz + GAP) + img_sz // 2
        bb = draw.textbbox((0, 0), col_labels[c], font=f_col)
        lw = bb[2] - bb[0]
        draw.text((cx - lw // 2, HEADER_H + 2), col_labels[c], fill=LABEL_COL, font=f_col)

    # Blit images
    arr = np.array(canvas)
    for c in range(3):
        px = x0 + c * (img_sz + GAP)
        if c == 2:
            img = imgs_a[2]  # GT constant
        else:
            a = imgs_a[c].astype(np.float32)
            b = imgs_b[c].astype(np.float32)
            img = ((1 - alpha) * a + alpha * b).astype(np.uint8)
        arr[y0:y0+img_sz, px:px+img_sz] = img

    canvas = Image.fromarray(arr)
    draw = ImageDraw.Draw(canvas)

    # Borders
    for c in range(3):
        px = x0 + c * (img_sz + GAP)
        draw.rectangle([px-1, y0-1, px+img_sz, y0+img_sz], outline=BORDER, width=1)

    # Photon meter
    frac = lerp(photon_frac(nf_a), photon_frac(nf_b), alpha)
    bar_w = meter_x1 - meter_x0

    draw.text((SIDE_PAD, meter_y + 1), "Photons", fill=LABEL_COL, font=f_meter_label)
    draw.rounded_rectangle([meter_x0, meter_y, meter_x1, meter_y + meter_h],
                           radius=5, fill=METER_BG)
    fw = max(int(bar_w * frac), 0)
    if fw > 3:
        # Gradient: lighter at low, darker at high
        t = frac
        fill = (int(41 + (26-41)*t), int(128 + (82-128)*t), int(185 + (118-185)*t))
        draw.rounded_rectangle([meter_x0, meter_y, meter_x0 + fw, meter_y + meter_h],
                               radius=5, fill=fill)

    # Ticks
    for _, _, tick_lbl, tick_nf in LEVELS:
        tf = photon_frac(tick_nf)
        tx = meter_x0 + int(bar_w * tf)
        draw.line([(tx, meter_y + meter_h), (tx, meter_y + meter_h + 7)],
                  fill=TICK_COL, width=1)
        short = tick_lbl.split("(")[0].strip().replace(" Filter", "").replace(" Frames", "f").replace(" Frame", "f")
        if "OD" in short:
            short = short  # keep "OD 0.7" etc
        bb = draw.textbbox((0, 0), short, font=f_meter_tick)
        lw = bb[2] - bb[0]
        draw.text((tx - lw // 2, meter_y + meter_h + 8), short, fill=TICK_COL, font=f_meter_tick)

    return canvas


# -- Build frame sequence ---------------------------------------------------
print("Building frame sequence...")

frames_forward = []  # list of BGR numpy arrays

n_levels = len(LEVELS)
n_scenes = len(SCENES)

for li in range(n_levels):
    _, _, label, nf = LEVELS[li]

    # Next level (for crossfade)
    if li < n_levels - 1:
        _, _, label_next, nf_next = LEVELS[li + 1]
    else:
        label_next, nf_next = label, nf

    # Show each scene for HOLD_FRAMES
    for si in range(n_scenes):
        imgs = resized[li][si]
        scene_name = SCENES[si]["name"]
        frame_pil = render(imgs, imgs, 0.0, label, nf, nf, scene_name)
        frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        for _ in range(HOLD_FRAMES):
            frames_forward.append(frame_bgr)

    # Crossfade to next level (using first scene as the transition visual)
    if li < n_levels - 1:
        si_fade = 0  # fade on first scene
        for fi in range(FADE_FRAMES):
            t = (fi + 1) / FADE_FRAMES
            alpha = 0.5 * (1 - math.cos(t * math.pi))
            imgs_a = resized[li][si_fade]
            imgs_b = resized[li+1][si_fade]
            lbl = label if alpha < 0.5 else label_next
            frame_pil = render(imgs_a, imgs_b, alpha, lbl, nf, nf_next,
                               SCENES[si_fade]["name"])
            frames_forward.append(cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR))

    print(f"  {label}: {len(frames_forward)} total frames")

# Reverse = forward frames in reverse (boomerang)
frames_reverse = frames_forward[::-1]

# Combine: forward + brief pause + reverse
pause_frames = [frames_forward[-1]] * int(FPS * 0.3)  # 0.3s pause at brightest

# -- Write videos -----------------------------------------------------------
def write_video(frames, path_raw, path_final):
    """Write frames to mp4v then re-encode to H.264."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path_raw), fourcc, FPS, (CANVAS_W, CANVAS_H))
    for f in frames:
        w.write(f)
    w.release()
    dur = len(frames) / FPS
    print(f"  Raw: {len(frames)} frames, {dur:.1f}s")

    try:
        cmd = ["ffmpeg", "-y", "-i", str(path_raw),
               "-c:v", "libx264", "-preset", "slow", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(path_final)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            path_raw.unlink()
            sz = path_final.stat().st_size / (1024*1024)
            print(f"  H.264: {path_final.name} ({sz:.1f} MB, {dur:.1f}s)")
        else:
            print(f"  ffmpeg failed, keeping mp4v")
            path_raw.rename(path_final)
    except Exception as e:
        print(f"  Re-encode error: {e}")
        if path_raw.exists():
            path_raw.rename(path_final)


tag = f"{HOLD_SEC:.1f}s_scene{SCENE_ID}".replace(".", "_")

print(f"\nWriting videos (hold={HOLD_SEC}s, tag={tag})...")
write_video(frames_forward,
            OUT_DIR / "photon_sweep_video.tmp.mp4",
            OUT_DIR / f"photon_sweep_forward_{tag}.mp4")

write_video(frames_reverse,
            OUT_DIR / "photon_sweep_reverse.tmp.mp4",
            OUT_DIR / f"photon_sweep_reverse_{tag}.mp4")

all_frames = frames_forward + pause_frames + frames_reverse
write_video(all_frames,
            OUT_DIR / "photon_sweep_video.tmp.mp4",
            OUT_DIR / f"photon_sweep_boomerang_{tag}.mp4")

print("\nDone!")
