"""Lineart track for a whole video: one mp4 in, one lineart mp4 out.

Run as a subprocess (`python -m diffsynth.utils.controlnet.video_lineart in.mp4
out.mp4`) when a generation pipeline follows: the detector sits on the GPU and
only process exit actually releases it. controlnet_aux and its cv2/skimage/timm
deps have to be importable, which may mean a PYTHONPATH the caller injects.
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .annotator import Annotator

DETECT_RESOLUTION = 512


def video_to_lineart(input_path, output_path, model_path="models/Annotators",
                     detect_resolution=DETECT_RESOLUTION, device="cuda"):
    """Extract lineart frame by frame, keeping the source fps and frame size.

    `detect_resolution` is the detector's internal short side; Annotator resizes
    its output back to each frame's own size. macro_block_size=1 keeps imageio
    from rounding the dimensions up to a multiple of 16 (1080 -> 1088).
    """
    annotator = Annotator("lineart", model_path=model_path,
                          detect_resolution=detect_resolution, device=device)
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()["fps"]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, quality=9, macro_block_size=1)
    n = 0
    for frame in reader:
        lineart = annotator(Image.fromarray(frame))
        writer.append_data(np.asarray(lineart.convert("RGB")))
        n += 1
    reader.close()
    writer.close()          # without close() the mp4 trailer is missing and the file is unreadable
    return output_path, n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--model-path", default="models/Annotators")
    p.add_argument("--detect-resolution", type=int, default=DETECT_RESOLUTION)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    path, n = video_to_lineart(args.input, args.output, model_path=args.model_path,
                               detect_resolution=args.detect_resolution, device=args.device)
    print(f"lineart: {n} frames -> {path}", flush=True)


if __name__ == "__main__":
    main()
