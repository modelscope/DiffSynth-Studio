"""
Paired SPAD Dataset for Frame Consistency Training.

For each sample, loads the GT RGB and TWO randomly-chosen single-frame binary
SPAD images from different frame folders (different temporal realizations of
the same scene). This enables the consistency loss:

    L_consistency = ||v_θ(z_t, t, F1) - v_θ(z_t, t, F2)||²

The two frames depict the same underlying scene so the denoised output should
be identical regardless of which binary frame is used as conditioning.
"""

import os
import re
import csv
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from diffsynth.core.data.operators import LoadImage, ImageCropAndResize


FRAME_FOLDERS = [
    ("bits",              "frames0-0"),
    ("bits_frame_1000",   "frames1000-1000"),
    ("bits_frame_4000",   "frames4000-4000"),
    ("bits_frame_5000",   "frames5000-5000"),
    ("bits_frame_8000",   "frames8000-8000"),
    ("bits_frame_12000",  "frames12000-12000"),
    ("bits_frame_16000",  "frames16000-16000"),
]


def _scene_id_from_path(rel_path: str) -> str | None:
    """Extract the scene identifier from a relative path like
    'bits/0724-dgp-001_RAW_empty_frames0-0_p.png' → '0724-dgp-001'."""
    fname = os.path.basename(rel_path)
    m = re.match(r"(.+?)_RAW_empty_frames", fname)
    return m.group(1) if m else None


def _build_frame_path(scene_id: str, folder: str, frame_tag: str) -> str:
    """Construct the relative path for a scene in a given frame folder."""
    return f"{folder}/{scene_id}_RAW_empty_{frame_tag}_p.png"


class PairedSPADDataset(torch.utils.data.Dataset):
    """Returns (GT_RGB, SPAD_F1, SPAD_F2) triplets for consistency training.

    F1 and F2 are randomly sampled from different frame folders each time
    ``__getitem__`` is called, providing data augmentation.
    """

    def __init__(
        self,
        base_path: str,
        metadata_csv: str,
        frame_folders: list[tuple[str, str]] | None = None,
        max_pixels: int = 262144,
        height: int | None = None,
        width: int | None = None,
        repeat: int = 1,
    ):
        self.base_path = Path(base_path)
        self.frame_folders = frame_folders or FRAME_FOLDERS
        self.repeat = repeat

        self.image_op = LoadImage(convert_RGB=True)
        self.resize_op = ImageCropAndResize(
            height=height, width=width, max_pixels=max_pixels,
            height_division_factor=16, width_division_factor=16,
        )

        with open(metadata_csv) as f:
            rows = list(csv.DictReader(f))

        self.load_from_cache = False

        self.samples = []
        for row in rows:
            ctrl_key = "controlnet_image" if "controlnet_image" in row else "input_image"
            scene_id = _scene_id_from_path(row[ctrl_key])
            if scene_id is None:
                continue

            available = []
            for folder, frame_tag in self.frame_folders:
                p = self.base_path / _build_frame_path(scene_id, folder, frame_tag)
                if p.exists():
                    available.append((folder, frame_tag))

            if len(available) < 2:
                continue

            self.samples.append({
                "scene_id": scene_id,
                "gt_path": str(row["image"]),
                "prompt": row.get("prompt", "") or "",
                "available_frames": available,
            })

        print(f"[PairedSPADDataset] {len(self.samples)} scenes with ≥2 frame folders "
              f"(from {len(rows)} total rows, {len(self.frame_folders)} folders)")

    def _load_and_resize(self, rel_path: str) -> Image.Image:
        abs_path = str(self.base_path / rel_path)
        img = self.image_op(abs_path)
        img = self.resize_op(img)
        return img

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        scene_id = sample["scene_id"]
        available = sample["available_frames"]

        f1_folder, f1_tag = random.choice(available)
        remaining = [x for x in available if x[0] != f1_folder]
        f2_folder, f2_tag = random.choice(remaining)

        f1_path = _build_frame_path(scene_id, f1_folder, f1_tag)
        f2_path = _build_frame_path(scene_id, f2_folder, f2_tag)

        gt_img = self._load_and_resize(sample["gt_path"])
        f1_img = self._load_and_resize(f1_path)
        f2_img = self._load_and_resize(f2_path)

        return {
            "image": gt_img,
            "controlnet_image": f1_img,
            "controlnet_image_f2": f2_img,
            "prompt": sample["prompt"],
        }
