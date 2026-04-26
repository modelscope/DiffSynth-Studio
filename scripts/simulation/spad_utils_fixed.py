"""
Race-free SPAD bit-unpacking utilities  —  fixed version.

The original `utils.py` in spad-diffusion has a data race in
`_accumulate_counts_core`: the Numba kernel parallelizes over frames with
`prange` but every iteration writes to the SAME `counts` array (e.g.
`counts[p + 0] += lut[0]`). Read-modify-write under prange is NOT atomic,
so concurrent updates lose increments.

This file provides drop-in replacements that fix the race using thread-local
accumulators followed by a reduction step. Same API:

    counts, n = accumulate_counts_whole_file(raw_bytes, max_frames, H, W)
    counts, n = accumulate_counts_frames(path, start, end, H, W, bitorder)

The `median_hotpixel_fix` helper is also re-exported (it had no race).

Also re-exports `_BIT_LUT` for downstream code that needs it.

Verified: produces deterministic, race-free results matching numpy ground truth.
"""
import numpy as np
from typing import Tuple, Optional

from numba import njit, prange
import numba

# cv2 is imported lazily inside median_hotpixel_fix() — keeps the module
# importable on systems without OpenCV (e.g. for code that only needs the
# count accumulator).


def _make_bit_lut() -> np.ndarray:
    """256 × 8 lookup table for MSB-first bit unpacking."""
    lut = np.empty((256, 8), dtype=np.uint8)
    for b in range(256):
        for i in range(8):
            lut[b, i] = (b >> (7 - i)) & 1
    return lut


_BIT_LUT = _make_bit_lut()


# -----------------------------------------------------------------------------
# Race-free core: per-thread chunk accumulators + serial reduction
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _accumulate_counts_core_safe(raw_bytes, max_frames, H, W, n_chunks):
    """
    Race-free version of the bit accumulation kernel.

    Parallelizes by splitting the frame range into n_chunks contiguous chunks.
    Each chunk gets its OWN counts array (chunk_counts[chunk_id, :]), so there
    are no concurrent writes to the same memory location. The final reduction
    is serial but trivially cheap (n_chunks × H*W typically ~16 × 262K = 4M).

    Args:
        raw_bytes: uint8 array, packed binary frames
        max_frames: maximum number of frames to accumulate
        H, W: image dimensions
        n_chunks: number of parallel chunks (typically = num_threads)

    Returns:
        (counts: HxW uint32, num_frames_processed: int)
    """
    bytes_per_frame = (H * W) // 8
    total_bytes = raw_bytes.size
    frames_avail = total_bytes // bytes_per_frame
    num_frames = min(max_frames, frames_avail)

    if n_chunks < 1:
        n_chunks = 1
    if n_chunks > num_frames:
        n_chunks = max(1, num_frames)

    chunk_size = (num_frames + n_chunks - 1) // n_chunks

    # Per-chunk accumulator: shape (n_chunks, H*W)  uint32
    # Memory: n_chunks × 262144 × 4 bytes = e.g. 16 × 1MB = 16 MB
    chunk_counts = np.zeros((n_chunks, H * W), dtype=np.uint32)

    for chunk_id in prange(n_chunks):
        start_f = chunk_id * chunk_size
        end_f = start_f + chunk_size
        if end_f > num_frames:
            end_f = num_frames
        for f in range(start_f, end_f):
            base = f * bytes_per_frame
            for k in range(bytes_per_frame):
                b = raw_bytes[base + k]
                p = k * 8
                lut = _BIT_LUT[b]
                chunk_counts[chunk_id, p + 0] += lut[0]
                chunk_counts[chunk_id, p + 1] += lut[1]
                chunk_counts[chunk_id, p + 2] += lut[2]
                chunk_counts[chunk_id, p + 3] += lut[3]
                chunk_counts[chunk_id, p + 4] += lut[4]
                chunk_counts[chunk_id, p + 5] += lut[5]
                chunk_counts[chunk_id, p + 6] += lut[6]
                chunk_counts[chunk_id, p + 7] += lut[7]

    # Serial reduction across chunks — race-free (only one thread)
    counts = np.zeros(H * W, dtype=np.uint32)
    for c in range(n_chunks):
        for p in range(H * W):
            counts[p] += chunk_counts[c, p]

    return counts.reshape(H, W), num_frames


# -----------------------------------------------------------------------------
# Public API (drop-in replacement)
# -----------------------------------------------------------------------------
def accumulate_counts_whole_file(raw_bytes: np.ndarray, max_frames: int,
                                  H: int = 512, W: int = 512
                                  ) -> Tuple[np.ndarray, int]:
    """
    Accumulate counts from raw binary bytes (race-free).

    Args:
        raw_bytes: uint8 array of packed binary frames
        max_frames: max frames to process
        H, W: image dimensions

    Returns:
        (counts: HxW uint32, num_frames_processed: int)
    """
    n_chunks = numba.get_num_threads()
    return _accumulate_counts_core_safe(raw_bytes, max_frames, H, W, n_chunks)


def accumulate_counts_frames(path: str, start_frame: int, end_frame: int,
                              H: int = 512, W: int = 512,
                              bitorder: str = "big") -> Tuple[np.ndarray, int]:
    """
    Stream-read frames [start, end) from a binary file and accumulate counts.

    Args:
        path: path to binary file
        start_frame, end_frame: frame range (half-open)
        H, W: image dimensions
        bitorder: 'big' or 'little'

    Returns:
        (counts: HxW uint32, num_frames_processed: int)
    """
    bytes_per_frame = (H * W) // 8

    frames_data = []
    frames_processed = 0
    try:
        with open(path, "rb") as f:
            for j in range(start_frame, end_frame):
                f.seek(j * bytes_per_frame)
                frame_bytes = f.read(bytes_per_frame)
                if len(frame_bytes) != bytes_per_frame:
                    break
                frames_data.append(frame_bytes)
                frames_processed += 1
    except FileNotFoundError:
        return np.zeros((H, W), dtype=np.uint32), 0

    if frames_processed == 0:
        return np.zeros((H, W), dtype=np.uint32), 0

    all_bytes = np.frombuffer(b"".join(frames_data), dtype=np.uint8)
    n_chunks = numba.get_num_threads()
    counts, _ = _accumulate_counts_core_safe(all_bytes, frames_processed, H, W, n_chunks)
    return counts, frames_processed


# -----------------------------------------------------------------------------
# Hot pixel fix (no race — kept identical to original utils.py)
# -----------------------------------------------------------------------------
def median_hotpixel_fix(img_counts: np.ndarray, num_frames: int, ksize: int = 3,
                         spike_thresh_abs: int = 10,
                         spike_thresh_rel: float = 0.01) -> np.ndarray:
    """Median-based hot pixel suppression on a counts array.

    Lazy-imports cv2 so that callers who never invoke this function don't pay
    the cv2 import cost or fail on environments without OpenCV.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "median_hotpixel_fix requires OpenCV (cv2). Install with "
            "`pip install opencv-python-headless` in the active environment."
        ) from e
    med = cv2.medianBlur(img_counts.astype(np.uint16), ksize)
    spike = (img_counts.astype(np.int32) - med.astype(np.int32)) > max(
        spike_thresh_abs, int(spike_thresh_rel * num_frames)
    )
    out = img_counts.copy()
    out[spike] = med[spike]
    return out


def read_file_bytes(path: str) -> Optional[np.ndarray]:
    """Read whole binary file as uint8 array, or None if missing."""
    try:
        with open(path, "rb") as f:
            b = f.read()
    except FileNotFoundError:
        return None
    return np.frombuffer(b, dtype=np.uint8)
