# MiniMax H3 integration constants, shared by the model code and the pipeline.

# ---------------------------------------------------------------------------
# Qwen3-VL presentation (ref: target presentation.py)
# ---------------------------------------------------------------------------
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"
# AdaLN modality tags: text rows are 1, Qwen vision spans are 0 (VIDEO).
PRESENTATION_TEXT_TAG = 1
PRESENTATION_VIDEO_TAG = 0
# Ref: target reference_encoding.py:502-503
QWEN_VIDEO_SAMPLE_FPS = 2.0
QWEN_TEMPORAL_PATCH = 2

# ---------------------------------------------------------------------------
# Canvas / spatial policy
# ---------------------------------------------------------------------------
# Ref: target resolved_plan.py:39-42
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32
# Ref: target reference_encoding.py:45-48
REFERENCE_IMAGE_SHORT_EDGE = 2048
REFERENCE_IMAGE_MULTIPLE = 32

# ---------------------------------------------------------------------------
# Temporal / VAE
# ---------------------------------------------------------------------------
# Ref: target constants.py:23
SUPPORTED_FPS = 24

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 32000
AUDIO_CHANNELS = 2

# ---------------------------------------------------------------------------
# Condition noise augmentation
# ---------------------------------------------------------------------------
# Each value drives both the mixed condition tensor and its AdaLN timestep, so
# moving off the trained anchor makes the condition out-of-distribution.
# Ref: target denoise_loop.py:21,23
IMGVID_COND_NOISE_AUG = 0.999
AUDIO_COND_NOISE_AUG = 1.0
