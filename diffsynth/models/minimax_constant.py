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
MINIMAX_H3_BASE_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
# Ref: target reference_encoding.py:45-48
MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE = 32

# ---------------------------------------------------------------------------
# Temporal / VAE
# ---------------------------------------------------------------------------
# Video VAE temporal chunking (config.json: vae_clip_length=17), 17n+5 grouping.
MINIMAX_H3_VAE_CLIP_LENGTH = 17
MINIMAX_H3_VAE_TAIL_FRAMES = 5
# Ref: target constants.py:23
MINIMAX_H3_SUPPORTED_FPS = 24

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_AUDIO_CHANNELS = 2
