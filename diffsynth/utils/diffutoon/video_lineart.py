"""Lineart track for a whole video: one mp4 in, one lineart mp4 out.

Call `video_to_lineart(input_path, output_path)` from Python. controlnet_aux and
its cv2/skimage/timm deps have to be importable, which may mean a PYTHONPATH the
caller injects. The detector runs on `device`; its GPU memory stays in PyTorch's
cache after the call returns, so run it in a separate process if the generation
pipeline that follows needs that memory back.

The detector weights are fetched through `ModelConfig`, so they land in the
framework's unified model directory (`./models` by default, overridable with
`DIFFSYNTH_MODEL_BASE_PATH`) instead of a machine-specific path.
"""
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from diffsynth.core import ModelConfig

from ..controlnet.annotator import Annotator

DETECT_RESOLUTION = 512
LINEART_MODEL_ID = "lllyasviel/Annotators"
LINEART_MODEL_PATTERN = "sk_model*.pth"


def resolve_annotator_path(model_path=None):
    """Directory holding the annotator weights, downloaded on first use.

    `model_path` is an optional local override. Otherwise `ModelConfig` downloads
    `LINEART_MODEL_PATTERN` from `LINEART_MODEL_ID` into the shared model directory
    and the containing directory is handed to `Annotator`.
    """
    if model_path is None:
        config = ModelConfig(model_id=LINEART_MODEL_ID, origin_file_pattern=LINEART_MODEL_PATTERN)
    else:
        config = ModelConfig(path=model_path)
    config.download_if_necessary()
    path = config.path
    if isinstance(path, list):
        path = path[0] if path else None
    if path is None:
        raise FileNotFoundError(
            f"{LINEART_MODEL_PATTERN} is missing; run without DIFFSYNTH_SKIP_DOWNLOAD to fetch "
            f"{LINEART_MODEL_ID} or pass model_path."
        )
    return path if os.path.isdir(path) else os.path.dirname(path)


def video_to_lineart(input_path, output_path, model_path=None,
                     detect_resolution=DETECT_RESOLUTION, device="cuda"):
    """Extract lineart frame by frame, keeping the source fps and frame size.

    `detect_resolution` is the detector's internal short side; Annotator resizes
    its output back to each frame's own size. macro_block_size=1 keeps imageio
    from rounding the dimensions up to a multiple of 16 (1080 -> 1088).
    """
    annotator = Annotator("lineart", model_path=resolve_annotator_path(model_path),
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
