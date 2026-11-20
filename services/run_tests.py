"""Run ``ModelService.run_tests`` for every inference service.

Usage (from the DiffSynth-Studio repository root):

    # List the registered services
    python -m services.run_tests --list

    # Run one service
    CUDA_VISIBLE_DEVICES=3 python -m services.run_tests --class QwenImage --repeats 1

    # Run every service, one subprocess per service (a crash cannot stop the suite)
    CUDA_VISIBLE_DEVICES=3 python -m services.run_tests --all --repeats 1

Each service runs as a separate process, so a model that fails to load or runs
out of memory does not abort the remaining tests.  Per-service logs and a JSON
summary are written to ``data/diffsynth_service_tests/logs``.
"""
import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback

# torch.compile needs Triton's Inductor backend.  If Triton is unavailable,
# disable Dynamo so the quantized deploy modes run in eager mode.  Set
# ALLOW_TORCH_COMPILE=1 to force compilation as deployed in production.
if os.environ.get("ALLOW_TORCH_COMPILE") != "1" and importlib.util.find_spec("triton") is None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    print("note: triton is unavailable; torch.compile falls back to eager (TORCHDYNAMO_DISABLE=1)", flush=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULT_DIR = os.path.join(ROOT, "data", "diffsynth_service_tests", "logs")

# (module, class name) in the order they should be tested.
SERVICES = [
    ("services.ace_step", "AceStep15XLSFT"),
    ("services.fibo", "Fibo20Dev"),
    ("services.fibo", "Fibo20EditTurbo"),
    ("services.flux2", "Flux2Klein4B"),
    ("services.flux2", "Flux2KleinBase4B"),
    ("services.flux2", "Flux2Klein9B"),
    ("services.flux2", "Flux2KleinBase9B"),
    ("services.hidream_o1", "HiDreamO1Image"),
    ("services.krea2_turbo", "Krea2Turbo"),
    ("services.minimax_h3", "MiniMaxH3FL2VA"),
    ("services.minimax_h3", "MiniMaxH3Ref2VA"),
    ("services.minimax_h3_image", "MiniMaxH3Image"),
    ("services.minimax_music", "MiniMaxMusic3"),
    ("services.moss_image", "MOSSImageSFT"),
    ("services.qwen_image", "QwenImage"),
    ("services.qwen_image", "QwenImage2512"),
    ("services.qwen_image", "QwenImageEdit"),
    ("services.qwen_image", "QwenImageEdit2509"),
    ("services.qwen_image", "QwenImageEdit2511"),
    ("services.qwen_image", "FireRedImageEdit"),
    ("services.qwen_image_21", "QwenImage21"),
    ("services.qwen_image_21", "QwenImage21Turbo"),
    ("services.z_image", "ZImageTurbo"),
    ("services.z_image", "ZImage"),
]

SERVICE_NAMES = [name for _, name in SERVICES]


def load_service(name):
    for module_name, class_name in SERVICES:
        if class_name == name:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)()
    raise ValueError(f"Unknown service: {name}. Use --list to see all services.")


def run_one(name, repeats, device, modes=None):
    service = load_service(name)
    started = time.time()
    try:
        service.run_tests(device=device, repeats=repeats, deploy_modes=modes)
        return {"service": name, "status": "success", "seconds": time.time() - started}
    except Exception:
        traceback.print_exc()
        return {"service": name, "status": "error", "seconds": time.time() - started}
    finally:
        service.pipe = None


def run_all(names, repeats, device, timeout, modes=None):
    os.makedirs(RESULT_DIR, exist_ok=True)
    summary = []
    for name in names:
        log_path = os.path.join(RESULT_DIR, f"{name}.log")
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        command = [sys.executable, "-m", "services.run_tests", "--class", name,
                   "--repeats", str(repeats), "--device", device]
        if modes:
            command += ["--modes", *[str(mode) for mode in modes]]
        with open(log_path, "w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
            )
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                code = 124
                log.write(f"\nTIMEOUT after {timeout} seconds\n")
        record = {
            "service": name,
            "status": "success" if code == 0 else "error",
            "exit_code": code,
            "log": os.path.relpath(log_path, ROOT),
        }
        print(f"{name}: {record['status']} (exit={code}, log={record['log']})", flush=True)
        summary.append(record)
        with open(os.path.join(RESULT_DIR, "summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list registered services")
    parser.add_argument("--class", dest="service", help="run a single service class")
    parser.add_argument("--all", action="store_true", help="run every service in a subprocess")
    parser.add_argument("--only", nargs="+", help="run a subset of service classes")
    parser.add_argument("--repeats", type=int, default=1, help="inference repeats per sample (default: 1)")
    parser.add_argument("--modes", nargs="+", type=int, help="deploy modes to test (default: all until NotImplementedError)")
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument("--timeout", type=int, default=6 * 3600, help="per-service timeout in seconds")
    args = parser.parse_args()

    if args.list:
        for module_name, class_name in SERVICES:
            print(f"{class_name:28s} {module_name}")
        return 0

    if args.service:
        record = run_one(args.service, args.repeats, args.device, args.modes)
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(os.path.join(RESULT_DIR, f"{args.service}.result.json"), "w") as handle:
            json.dump(record, handle, indent=2)
        return 0 if record["status"] == "success" else 1

    names = args.only or SERVICE_NAMES
    if not (args.all or args.only):
        parser.error("choose --list, --class, or --all/--only")
    summary = run_all(names, args.repeats, args.device, args.timeout, args.modes)
    failed = [item["service"] for item in summary if item["status"] != "success"]
    print(f"\n{'=' * 70}")
    print(f"finished: {len(summary) - len(failed)}/{len(summary)} services succeeded")
    if failed:
        print("failed:", ", ".join(failed))
    print(f"summary: {os.path.join(RESULT_DIR, 'summary.json')}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
