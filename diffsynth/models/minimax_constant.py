# MiniMax H3 integration constants, shared by the model code and the pipeline.
# ---------------------------------------------------------------------------
# Canvas / spatial policy
# ---------------------------------------------------------------------------
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32

REFERENCE_IMAGE_SHORT_EDGE = 2048
REFERENCE_IMAGE_MULTIPLE = 32
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
IMGVID_COND_NOISE_AUG = 0.999
AUDIO_COND_NOISE_AUG = 1.0
