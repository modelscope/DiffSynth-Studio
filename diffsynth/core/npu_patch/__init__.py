from diffsynth.core.device.npu_compatible_device import IS_NPU_AVAILABLE
from .npu_autocast_patch import npu_autocast_patch

if IS_NPU_AVAILABLE:
    npu_autocast_patch()
