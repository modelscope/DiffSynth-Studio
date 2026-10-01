"""VRAM residency guard for GPUs shared with a desktop session.

`vram_limit` in VRAM management is a residency threshold: layers stop being
kept on the GPU once device-wide usage crosses it, but transient layers,
activations and the caching allocator still grow above it. On a card the
desktop shares, growth to the physical limit forces driver eviction under
running kernels. The guard adds the two missing bounds:

- a hard allocator cap (`torch.cuda.set_per_process_memory_fraction`), so an
  overcommit becomes a clean OutOfMemoryError instead of a driver fault;
- an abort gate in a separate process that SIGKILLs this process when the
  card's sysfs `mem_info_vram_used` crosses the abort line (backstop for
  memory the caching allocator does not see).
"""
import glob, os, signal, subprocess, sys, time

import torch

GIB = 1024 ** 3

_WATCHDOG = r"""
import os, signal, sys, time
pid, sysfs, limit, peak_file = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
peak = 0
while True:
    try:
        os.kill(pid, 0)
    except OSError:
        break
    used = int(open(sysfs).read())
    if used > peak:
        peak = used
        open(peak_file, "w").write(str(peak))
    if used > limit:
        open(peak_file, "w").write(str(peak) + " ABORTED")
        os.kill(pid, signal.SIGKILL)
        break
    time.sleep(0.02)
"""


def find_sysfs_vram(device=0):
    # The card whose sysfs total matches the torch device total.
    total = torch.cuda.get_device_properties(device).total_memory
    best = None
    for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
        delta = abs(int(open(path).read()) - total)
        if best is None or delta < best[0]:
            best = (delta, path.replace("_total", "_used"))
    if best is None or best[0] > 0.02 * total:
        raise RuntimeError("No amdgpu sysfs VRAM counter matches the torch device.")
    return best[1]


def sysfs_vram_used_gib(sysfs):
    return int(open(sysfs).read()) / GIB


def amdgpu_fault_lines():
    # Current boot only; journalctl reads the kernel ring without root.
    try:
        out = subprocess.run(["journalctl", "-k", "-b", "0", "--no-pager", "-o", "short-monotonic"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        out = subprocess.run(["dmesg"], capture_output=True, text=True).stdout
    keys = ("page fault", "timeout", "GPU reset", "reset failed", "PROTECTION_FAULT", "Memory Fault")
    return [l for l in out.splitlines() if "amdgpu" in l and any(k in l for k in keys)]


def host_peak_rss_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024 / GIB
    return 0.0


def check_rocm_allocator():
    # torch 2.13.0+rocm10.0 / HIP 7.15 (gfx1201): with expandable segments,
    # empty_cache() during VRAM-managed inference corrupts weights that are
    # still resident (LTX-2.5 Gemma encoder: NaN in 7/8 passes vs 0/8 with
    # the default allocator, same config), and kernels reading released
    # pages fault (amdgpu gfxhub permission faults). VRAM management calls
    # empty_cache() on every model switch, so refuse the combination.
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") + "," + os.environ.get("PYTORCH_ALLOC_CONF", "")
    if torch.version.hip and "expandable_segments:true" in conf.replace(" ", "").lower():
        if os.environ.get("DIFFSYNTH_ALLOW_ROCM_EXPANDABLE_SEGMENTS") != "1":
            raise RuntimeError(
                "expandable_segments:True corrupts resident weights across empty_cache() on this ROCm stack; "
                "unset PYTORCH_CUDA_ALLOC_CONF (or set DIFFSYNTH_ALLOW_ROCM_EXPANDABLE_SEGMENTS=1 to override)."
            )


def route_rocm_blas(log=print):
    # On gfx1201 (torch 2.13.0+rocm10.0) every bf16 GEMM, including rocBLAS
    # (which forwards to hipBLASLt by default) and the fused-bias addmm path,
    # dispatches hipBLASLt "UserArgs" kernels; under offloaded LoRA training
    # these read unmapped pages (amdgpu gfxhub permission faults, 5/5 runs).
    # rocBLAS's own Tensile kernels ran the same training clean. Must run
    # before the first GEMM: rocBLAS reads the switch at handle creation.
    if not torch.version.hip or os.environ.get("DIFFSYNTH_ROCM_BLAS") == "hipblaslt":
        return
    os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "0")
    torch.backends.cuda.preferred_blas_library("cublas")
    log(f"BLAS_ROUTE=rocblas-tensile ROCBLAS_USE_HIPBLASLT={os.environ['ROCBLAS_USE_HIPBLASLT']} "
        f"(set DIFFSYNTH_ROCM_BLAS=hipblaslt to opt out)")


class VRAMGuard:
    def __init__(self, abort_gib, device=0, cap_margin_gib=0.5, log=print):
        check_rocm_allocator()
        route_rocm_blas(log)
        self.abort_gib = abort_gib
        self.device = device
        self.log = log
        self.sysfs = find_sysfs_vram(device)
        self.total_gib = torch.cuda.get_device_properties(device).total_memory / GIB
        self.baseline_gib = sysfs_vram_used_gib(self.sysfs)
        self.faults_before = len(amdgpu_fault_lines())
        # Process cap: everything this process may allocate while others
        # (desktop) keep their baseline, with a margin for non-allocator use
        # (HIP runtime, kernels, hipBLASLt handles).
        self.cap_gib = abort_gib - self.baseline_gib - cap_margin_gib
        if self.cap_gib <= 1:
            raise RuntimeError(f"Abort line {abort_gib} GiB leaves no room above baseline {self.baseline_gib:.2f} GiB.")
        torch.cuda.set_per_process_memory_fraction(self.cap_gib / self.total_gib, device)
        self.peak_file = f"/tmp/vram_guard_{os.getpid()}.peak"
        open(self.peak_file, "w").write("0")
        self.proc = subprocess.Popen([sys.executable, "-c", _WATCHDOG, str(os.getpid()), self.sysfs,
                                      str(int(abort_gib * GIB)), self.peak_file], start_new_session=True)
        self.log(f"ABORT_GATE=ARMED limit={abort_gib:.2f}GB sysfs={self.sysfs} "
                 f"baseline={self.baseline_gib:.2f}GB process_cap={self.cap_gib:.2f}GB "
                 f"faults_before={self.faults_before}")

    def phase(self, name):
        self.log(f"VRAM_PHASE {name} sysfs={sysfs_vram_used_gib(self.sysfs):.2f}GB "
                 f"alloc={torch.cuda.memory_allocated(self.device) / GIB:.2f}GB "
                 f"reserved={torch.cuda.memory_reserved(self.device) / GIB:.2f}GB "
                 f"peak_sysfs={self.peak_gib():.2f}GB")

    def peak_gib(self):
        try:
            return float(open(self.peak_file).read().split()[0]) / GIB
        except Exception:
            return float("nan")

    def report(self):
        time.sleep(0.1)
        fault_lines = amdgpu_fault_lines()
        new_faults = fault_lines[self.faults_before:]
        self.log(f"PEAK_VRAM_GB={self.peak_gib():.2f} HOST_RSS_GB={host_peak_rss_gib():.2f} "
                 f"VRAM_BEFORE_GB={self.baseline_gib:.2f} VRAM_AFTER_GB={sysfs_vram_used_gib(self.sysfs):.2f}")
        self.log("AMDGPU_FAULTS=NONE" if not new_faults else f"AMDGPU_FAULTS={len(new_faults)}")
        for line in new_faults[:20]:
            self.log("  " + line)
        return not new_faults

    def close(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
        try:
            os.remove(self.peak_file)
        except OSError:
            pass
