import atexit
import os
import sys
import threading
import time
import torch


IS_WINDOWS_ROCM = sys.platform == "win32" and torch.version.hip is not None

# ADLX takes about half a millisecond per query and `check_free_vram` runs on every layer forward,
# so the VRAM used by other processes is refreshed at most this often (seconds).
OTHER_PROCESSES_REFRESH = 1.0

if torch.version.hip is not None:
    # Without it, PyTorch SDPA has no flash or memory-efficient kernel on the GPUs AOTriton still
    # marks experimental and falls back to the math path. Read at the first SDPA call.
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

_adlx = None
_adlx_lock = threading.Lock()
_other_processes = {}
_missing_adlx_tips_printed = False


def _adlx_session():
    global _adlx, _missing_adlx_tips_printed
    with _adlx_lock:
        if _adlx is None:
            _adlx = {}
            try:
                from adlx import ADLX
            except ImportError:
                if not _missing_adlx_tips_printed:
                    print("On ROCm for Windows, torch only reports the VRAM used by this process. Install `amd-adlx` so that `vram_limit` also accounts for other programs and the desktop.")
                    _missing_adlx_tips_printed = True
                return None
            try:
                helper = ADLX.ADLXHelper()
                if helper.Initialize() == ADLX.ADLX_RESULT.ADLX_OK:
                    system = helper.GetSystemServices()
                    # UniqueId packs the PCI location as (bus << 8) | (device << 3) | function
                    gpus = {(gpu.UniqueId() >> 8) & 0xFF: gpu for gpu in system.GetGPUs()}
                    _adlx.update(helper=helper, perf=system.GetPerformanceMonitoringServices(), gpus=gpus)
                    atexit.register(_adlx_close)
            except Exception as e:
                print(f"ADLX is unavailable ({e}), falling back to the VRAM torch reports for this process.")
        return _adlx or None


def _adlx_close():
    # ADLX crashes at interpreter exit unless every interface is released before Terminate.
    with _adlx_lock:
        helper = _adlx.pop("helper")
        _adlx.clear()
        helper.Terminate()


def _adlx_used_bytes(bus):
    session = _adlx_session()
    if session is None or bus not in session["gpus"]:
        return None
    with _adlx_lock:
        metrics = session["perf"].GetCurrentGPUMetrics(session["gpus"][bus])
        used = metrics.GPUVRAM()
        del metrics
    return used * 1024 ** 2


def mem_get_info(device=None):
    """`torch.cuda.mem_get_info`, but for the whole GPU on ROCm for Windows.

    There torch only counts its own process, so the desktop and other programs look like free VRAM.
    The VRAM they use comes from ADLX instead.
    """
    free, total = torch.cuda.mem_get_info(device)
    if not IS_WINDOWS_ROCM:
        return free, total
    used_by_this_process = total - free
    bus = torch.cuda.get_device_properties(device).pci_bus_id
    now = time.monotonic()
    cached = _other_processes.get(bus)
    if cached is None or now - cached[0] > OTHER_PROCESSES_REFRESH:
        used = _adlx_used_bytes(bus)
        if used is None:
            return free, total
        cached = _other_processes[bus] = (now, max(used - used_by_this_process, 0))
    return max(total - used_by_this_process - cached[1], 0), total
