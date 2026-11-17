#!/usr/bin/env python3
"""Collect environment facts; framework imports and Torch/device queries are opt-in subprocesses."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlparse


def redact(text, *, replace_environment_values=True):
    """Best-effort redaction of known credential values and common credential syntax."""
    text = text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text or "")
    if replace_environment_values:
        secrets = {value for key, value in os.environ.items()
                   if len(value) >= 4 and re.search(r"TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|AUTHORIZATION|CREDENTIAL|ACCESS[_-]?KEY", key, re.I)}
        for value in sorted(secrets, key=len, reverse=True):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(?i)(https?://)[^\s/@]+@", r"\1[REDACTED]@", text)
    field = r'''\b(?:[a-z0-9]+[_-])*(?:token|secret|password|passwd|api[_-]?key|authorization|credential|signature)(?:[_-][a-z0-9]+)*["']?\s*[:=]\s*'''
    text = re.sub(r'''(?is)('''+field+r''')(["'])((?:\\.|(?!\2)[^\\])*)\2''', r"\1\2[REDACTED]\2", text)
    text = re.sub(r"(?im)(\b(?:[a-z0-9]+[_-])*authorization\s*[:=]\s*)[^\r\n]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
    return re.sub(r'''(?i)('''+field+r''')([^\s,;&"'<>]+)''',
                  r"\1[REDACTED]", text)


def redact_report(value, field=None, preserve_literals=False):
    """Redact diagnostic/configuration text without rewriting structured facts."""
    protected = {
        "status", "stage", "error_type", "code", "device", "name", "kind",
        "scope", "state_timing", "memory_architecture", "backend",
        "version", "python_version", "requires_python", "observed_at", "commit",
        "device_identifier_hash", "requested_source_match", "distribution_origin_match",
        "python_requirement", "package_import", "device_query", "tensor_op", "backend_op",
        "pipeline", "target_readiness",
    }
    if field and re.search(r"(?i)(?:^|[_-])(?:token|secret|password|passwd|api[_-]?key|authorization|credentials?|access[_-]?key)(?:$|[_-])", field):
        return "[REDACTED]"
    if isinstance(value, str):
        literal = preserve_literals or field in protected or bool(field and field.endswith(("_status", "_version")))
        # Credential syntax/URLs are still filtered even in structured fields.
        return redact(value, replace_environment_values=not literal)
    if isinstance(value, dict):
        return {key: redact_report(item, key, preserve_literals=field == "packages")
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_report(item, field, preserve_literals) for item in value]
    return value


def issue(stage, error, **context):
    return dict(stage=stage, error_type=type(error).__name__, message=redact(str(error))[-4096:], **context)


def git_info(root):
    """Only attribute Git state to the requested checkout, never its parent repo."""
    info = {"observed_at": datetime.now(timezone.utc).isoformat(),
            "state_timing": "before_output_write",
            "status": "unavailable", "root": None, "matches_source_root": False,
            "commit": None, "dirty": None}

    def run(*args):
        return subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True, timeout=10)

    try:
        top = run("rev-parse", "--show-toplevel")
        if top.returncode:
            return info
        info["root"] = str(Path(top.stdout.strip()).resolve())
        if Path(info["root"]) != root.resolve():
            info["status"] = "parent_worktree"
            return info
        info["matches_source_root"] = True
        head = run("rev-parse", "HEAD")
        state = run("status", "--porcelain", "--untracked-files=normal")
        info["commit"] = head.stdout.strip() if head.returncode == 0 else None
        info["dirty"] = bool(state.stdout) if state.returncode == 0 else None
        info["status"] = "available" if info["commit"] and info["dirty"] is not None else "incomplete"
    except (OSError, ValueError, subprocess.TimeoutExpired):
        # Exported sources and minimal environments may have no usable Git.
        pass
    return info


def storage_info(path):
    """Probe the target filesystem without creating a missing target directory."""
    if path is None:
        return {"status": "not_specified", "path": None}
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise ValueError("Storage paths must be absolute; resolve them from the actual task cwd, not the probe cwd")
    target = target.resolve()
    existing = target
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    return {"status": "ok", "path": str(target), "target_exists": target.exists(),
            "probed_path": str(existing), "quota_checked": False,
            "filesystem_device_id": existing.stat().st_dev,
            **dict(zip(("total_bytes", "used_bytes", "free_bytes"), usage))}


def cgroup_memory(proc):
    """Read the process's visible v1/v2 memory hierarchy, including parent limits."""
    result = {"status": "not_detected", "version": None, "levels": [],
              "limit_bytes": None, "headroom_bytes": None, "errors": []}
    try:
        memberships = [line.split(":", 2) for line in (proc/'self/cgroup').read_text().splitlines()]
        memory_on_v1 = any("memory" in controllers.split(",") for _, controllers, _ in memberships)
        mounts = (proc/'self/mountinfo').read_text().splitlines()
        candidates = []
        for line in mounts:
            before, after = line.split(" - ", 1)
            fields, fs = before.split(), after.split()
            if fs[0] not in ("cgroup", "cgroup2"):
                continue
            version = 2 if fs[0] == "cgroup2" else 1
            if version == 2 and memory_on_v1:
                continue  # In hybrid mode the memory controller belongs to v1, not unified.
            if version == 1 and "memory" not in fs[2].split(","):
                continue
            decode = lambda text: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), text)
            mount_root, mount = PurePosixPath(decode(fields[3])), Path(decode(fields[4]))
            for _, controllers, group in memberships:
                if not ((version == 2 and not controllers) or (version == 1 and "memory" in controllers.split(","))):
                    continue
                group = PurePosixPath(group)
                if ".." in group.parts:
                    continue  # Cannot safely map a namespace-external path onto this mount.
                if group.is_relative_to(mount_root):
                    relative = group.relative_to(mount_root)
                elif group == PurePosixPath("/"):
                    relative = PurePosixPath(".")  # Namespace maps its root to this mount.
                else:
                    continue
                candidates.append((len(mount_root.parts), version, mount, mount/str(relative)))
        if not candidates:
            if any(not controllers or "memory" in controllers.split(",") for _, controllers, _ in memberships):
                result["status"] = "partial"
                result["errors"].append(issue("cgroup_memory", ValueError("Cannot map the process memory cgroup to a readable mount")))
            return result
        # A broader readable mount exposes more ancestors than a subtree bind mount.
        _, version, mount, current = min(candidates, key=lambda item: item[0])
        result.update(status="ok", version=version, mount=str(mount))
        limit_name, usage_name = (("memory.max", "memory.current") if version == 2 else
                                  ("memory.limit_in_bytes", "memory.usage_in_bytes"))
        while True:
            limit_file, usage_file = current/limit_name, current/usage_name
            if limit_file.exists():
                try:
                    raw = limit_file.read_text().strip()
                    limit = None if raw == "max" else int(raw)
                    if version == 1 and limit is not None and limit >= 2**60:
                        limit = None  # v1's conventional unlimited sentinel.
                    usage = int(usage_file.read_text().strip())
                    if usage < 0 or (limit is not None and limit < 0):
                        raise ValueError("Negative cgroup memory value")
                    result["levels"].append({"path": str(current), "limit_bytes": limit,
                                             "usage_bytes": usage,
                                             "headroom_bytes": max(0, limit-usage) if limit is not None else None})
                except (OSError, ValueError) as error:
                    result["status"] = "partial"
                    result["errors"].append(issue("cgroup_memory", error, path=str(current)))
            elif current != mount or version == 1:
                # A v2 root legitimately has no memory.max; other missing controls are unknown.
                result["status"] = "partial"
                result["errors"].append(issue("cgroup_memory", FileNotFoundError(str(limit_file))))
            if current == mount:
                break
            current = current.parent
        finite = [level for level in result["levels"] if level["limit_bytes"] is not None]
        if finite:
            result["limit_bytes"] = min(level["limit_bytes"] for level in finite)
            result["headroom_bytes"] = min(level["headroom_bytes"] for level in finite)
    except FileNotFoundError:
        result["status"] = "unavailable"
    except (OSError, ValueError, IndexError) as error:
        result["status"] = "partial"
        result["errors"].append(issue("cgroup_memory", error))
    return result


def memory_info(proc=Path("/proc")):
    host = {"status": "unavailable", "total_bytes": None, "available_bytes": None}
    errors = []
    try:
        entries = dict(line.split(":", 1) for line in (proc/'meminfo').read_text().splitlines())
        host.update(status="ok", total_bytes=int(entries["MemTotal"].split()[0])*1024,
                    available_bytes=int(entries["MemAvailable"].split()[0])*1024)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError) as error:
        errors.append(issue("host_memory", error))
    group = cgroup_memory(proc)
    available = [value for value in (host["available_bytes"], group["headroom_bytes"]) if value is not None]
    known = group["status"] == "ok" and bool(group["levels"])
    group["limit_status"] = ("finite" if group["limit_bytes"] is not None else "no_finite_limit_observed") if known else "unknown"
    approximate = min(available) if known and available else None
    effective_status = ("unknown_cgroup_limit" if not known else
                        "unavailable" if approximate is None else
                        "cgroup_headroom_only" if host["available_bytes"] is None else "estimated")
    return {"host_visible": host, "cgroup": group, "approx_available_bytes": approximate,
            "effective_memory_status": effective_status,
            "scope": "Snapshot of host-visible RAM and readable cgroup ancestors; not reserved memory. Hidden ancestors, other job limits and reclaimable memory are not resolved.",
            "errors": errors+group["errors"]}


def metadata_info(version, source_root):
    result = {"status": "not_found", "spec_origin": None, "distribution_origin": None,
              "distribution_origin_match": "unknown", "requested_source_match": "not_requested",
              "import_verified": False, "errors": []}
    try:
        spec = importlib.util.find_spec("diffsynth")
        result["spec_origin"] = spec.origin if spec else None
        found = spec is not None
        result["status"] = ("distribution_and_spec_found" if version is not None and found else
                            "metadata_only" if version is not None else "spec_only" if found else "not_found")
        if version is not None:
            distribution = importlib.metadata.distribution("diffsynth")
            direct = distribution.read_text("direct_url.json")
            editable = json.loads(direct) if direct else {}
            if not isinstance(editable, dict) or not isinstance(editable.get("dir_info", {}), dict):
                raise ValueError("direct_url.json and its dir_info must be JSON objects")
            url = urlparse(editable.get("url", ""))
            if editable.get("dir_info", {}).get("editable") and url.scheme == "file" and url.netloc in ("", "localhost"):
                expected = Path(unquote(url.path))/'diffsynth/__init__.py'
            else:
                expected = Path(distribution.locate_file('diffsynth/__init__.py'))
            if expected.is_file():
                result["distribution_origin"] = str(expected.resolve())
                if result["spec_origin"]:
                    result["distribution_origin_match"] = "match" if Path(result["spec_origin"]).resolve() == expected.resolve() else "mismatch"
    except (ImportError, OSError, ValueError, TypeError, AttributeError) as error:
        result["status"] = "discovery_failed"
        result["errors"].append(issue("metadata_discovery", error))
    if source_root:
        result["requested_source_match"] = "unknown" if result["status"] == "discovery_failed" else "not_found"
        if result["spec_origin"]:
            try:
                expected = Path(source_root).expanduser().resolve()/'diffsynth/__init__.py'
                result["requested_source_match"] = "match" if expected.is_file() and Path(result["spec_origin"]).resolve() == expected else "mismatch"
            except (OSError, ValueError, RuntimeError) as error:
                result["requested_source_match"] = "unknown"
                result["errors"].append(issue("source_origin_comparison", error))
    return result


def torch_state(requested):
    return {"torch_probe_requested": requested, "torch_import_ok": None, "device_query_ok": None,
            "tensor_op_verified": False, "device_probe_status": "not_requested",
            "devices": [], "errors": [],
            "device_probe_scope": "Process-visible CUDA/ROCm and MPS only; NPU needs its vendor environment. No tensor operation is performed."}


def vram_budget(free_bytes, total_bytes, reserve_gib):
    """Separate new-task headroom from DiffSynth's device-used-memory threshold."""
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("reserve_gib must be finite and nonnegative")
    if not (0 <= free_bytes <= total_bytes):
        raise ValueError("Expected 0 <= free_bytes <= total_bytes")
    available = free_bytes / 1024**3 - reserve_gib
    return {"status": "estimated" if available > 0 else "no_weight_budget",
            "reserve_gib": reserve_gib, "new_task_budget_gib": max(0, available),
            "vram_limit_gib": max(0, total_bytes / 1024**3 - reserve_gib),
            "note": "Snapshot only. Reserve must cover task activations, workspaces and transient weights; not a measured peak or resource reservation."}


def weight_file_info(path):
    """Read local safetensors headers only; never materialize weights or import torch."""
    path = Path(path).expanduser().resolve()
    row = {"path": str(path), "status": "unavailable"}
    try:
        if path.suffix != ".safetensors":
            raise ValueError("Header inspection supports .safetensors only; binary weights are not loaded")
        file_bytes = path.stat().st_size
        with path.open("rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            if not 2 <= length <= min(64 * 1024**2, file_bytes - 8):
                raise ValueError("Invalid or oversized safetensors header (limit: 64 MiB)")
            header = json.loads(handle.read(length))
        if not isinstance(header, dict):
            raise ValueError("Invalid safetensors header")
        dtype_bytes = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
                       "U32": 4, "I32": 4, "U64": 8, "I64": 8,
                       "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
                       "F8_E4M3": 1, "F8_E5M2": 1}
        stored_numel, tensor_bytes, largest, matrix_numel = 0, 0, 0, 0
        by_dtype, ranges = {}, []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            shape, dtype, offsets = tensor["shape"], tensor["dtype"], tensor["data_offsets"]
            if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                    or not isinstance(dtype, str) or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)
                    or not 0 <= offsets[0] <= offsets[1] <= file_bytes - 8 - length):
                raise ValueError("Invalid tensor shape, dtype or data offsets")
            count, size = math.prod(shape), offsets[1] - offsets[0]
            if dtype in dtype_bytes and size != count * dtype_bytes[dtype]:
                raise ValueError("Tensor data size does not match shape and dtype")
            stored_numel += count
            tensor_bytes += size
            largest = max(largest, size)
            by_dtype[dtype] = by_dtype.get(dtype, 0) + count
            if dtype in ("BF16", "F16", "F32", "F64") and len(shape) == 2 and (name == "weight" or name.endswith(".weight")):
                matrix_numel += count
            if size:
                ranges.append(offsets)
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError("Overlapping tensor data offsets")
        row.update(status="header_read", file_bytes=file_bytes, stored_numel=stored_numel,
                   tensor_data_bytes=tensor_bytes, largest_tensor_bytes=largest, numel_by_dtype=by_dtype,
                   floating_2d_weight_numel=matrix_numel,
                   note="Stored tensor elements, not logical parameter count for packed checkpoints. 2D weights include embeddings; verify nn.Linear targets, converters, exclusions and complete shards before estimating quantization.")
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        row["error"] = str(error)
    return row


def memory_architecture(backend, properties=None):
    """Report shared physical memory only from an explicit architecture signal."""
    if backend == "mps" and platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64"):
        return {"kind": "unified", "cpu_offload_useful": False, "evidence": "Apple Silicon MPS on ARM macOS"}
    for name in ("is_integrated", "integrated"):
        value = getattr(properties, name, None)
        if type(value) in (bool, int) and value in (0, 1):
            return {"kind": "unified" if value else "discrete", "cpu_offload_useful": not bool(value),
                    "evidence": "Device property " + name + "=" + str(value)}
    return {"kind": "unknown", "cpu_offload_useful": None,
            "evidence": "Physical memory topology not reported; unified addressing or managed-memory support alone is insufficient"}


def probe_torch():
    data = torch_state(True)
    try:
        import torch
    except Exception as error:
        data.update(torch_import_ok=False, device_query_ok=False, device_probe_status="import_failed")
        data["errors"].append(issue("torch_import", error))
        return data
    data.update(torch_import_ok=True, torch_import_origin=torch.__file__,
                torch_build_cuda_version=getattr(torch.version, "cuda", None),
                torch_build_hip_version=getattr(torch.version, "hip", None))
    try:
        data["cuda_available"] = bool(torch.cuda.is_available())
        if data["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                device = {"device": f"cuda:{index}", "status": "ok", "tensor_op_verified": False,
                          "memory_architecture": memory_architecture("cuda")}
                try:
                    device["name"] = torch.cuda.get_device_name(index)
                    device["free_bytes"], device["total_bytes"] = torch.cuda.mem_get_info(index)
                except Exception as error:
                    device["status"] = "failed"
                    data["errors"].append(issue("cuda_device_query", error, device=device["device"]))
                if device["status"] == "ok":
                    try:
                        device["compute_capability"] = list(torch.cuda.get_device_capability(index))
                    except Exception as error:
                        device["compute_capability"] = None
                        data["errors"].append(issue("cuda_capability_query", error, device=device["device"]))
                    properties = getattr(torch.cuda, "get_device_properties", None)
                    if properties is not None:
                        try:
                            device_properties = properties(index)
                            device["memory_architecture"] = memory_architecture("cuda", device_properties)
                            identifier = getattr(device_properties, "uuid", None)
                            if identifier is not None:
                                import hashlib
                                device["device_identifier_hash"] = hashlib.sha256(str(identifier).encode()).hexdigest()
                        except Exception as error:
                            data["errors"].append(issue("cuda_identifier_query", error, device=device["device"]))
                data["devices"].append(device)
    except Exception as error:
        data["errors"].append(issue("cuda_availability", error))
    try:
        mps = getattr(torch.backends, "mps", None)
        data["mps"] = {"is_built": bool(mps and mps.is_built()),
                       "available": bool(mps and mps.is_available()), "tensor_op_verified": False}
        if data["mps"]["available"]:
            data["devices"].append({"device": "mps", "status": "ok", "tensor_op_verified": False,
                                    "memory_architecture": memory_architecture("mps")})
    except Exception as error:
        data["errors"].append(issue("mps_query", error))
    data["device_query_ok"] = not data["errors"]
    data["device_probe_status"] = "partial" if data["errors"] else "ok" if data["devices"] else "no_accelerator"
    return data


def probe_import():
    try:
        import diffsynth
        return {"status": "pass", "origin": diffsynth.__file__, "errors": []}
    except Exception as error:
        return {"status": "fail", "origin": None, "errors": [issue("diffsynth_import", error)]}


def worker_diagnostics(function, returncode=None, stdout=None, stderr=None, timed_out=False):
    return {"stage": function, "returncode": returncode,
            "signal": -returncode if os.name == "posix" and returncode is not None and returncode < 0 else None,
            "timed_out": timed_out, "stdout_tail": redact(stdout)[-4096:],
            "stderr_tail": redact(stderr)[-4096:]}


def run_worker(function, timeout):
    # Preserve this interpreter's search path and isolate import side effects/output.
    code = ('import contextlib,json,runpy,sys\n'
            'sys.path=json.loads(sys.argv[2])\n'
            'with contextlib.redirect_stdout(sys.stderr):\n'
            '    namespace=runpy.run_path(sys.argv[1])\n'
            '    result=namespace[sys.argv[3]]()\n'
            'print("DIFFSYNTH_PROBE_JSON="+json.dumps(result))\n')
    child = None
    try:
        child = subprocess.run([sys.executable, '-c', code, str(Path(__file__).resolve()),
                                json.dumps(sys.path), function], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout)
        if child.returncode:
            raise RuntimeError(f"Probe subprocess exited with code {child.returncode}")
        result = json.loads(child.stdout.rsplit("DIFFSYNTH_PROBE_JSON=", 1)[1])
        if not isinstance(result, dict):
            raise ValueError("Probe subprocess response must be a JSON object")
        return {"status": "ok", "result": result, "errors": [],
                "diagnostics": worker_diagnostics(function, child.returncode,
                    child.stdout.split("DIFFSYNTH_PROBE_JSON=", 1)[0], child.stderr)}
    except subprocess.TimeoutExpired as error:
        return {"status": "timeout", "errors": [{"stage": function, "error_type": type(error).__name__,
                "message": f"Probe subprocess exceeded {timeout} seconds"}],
                "diagnostics": worker_diagnostics(function, stdout=error.stdout, stderr=error.stderr, timed_out=True)}
    except (OSError, ValueError, IndexError, RuntimeError) as error:
        return {"status": "failed", "errors": [issue(function, error)],
                "diagnostics": worker_diagnostics(function, child.returncode if child is not None else None,
                    child.stdout if child is not None else None, child.stderr if child is not None else None)}


def source_validation(root):
    if not root:
        return {"status": "not_requested", "path": None, "markers": []}
    path = Path(root).expanduser().resolve()
    result = {"status": "source_invalid", "path": str(path), "markers": []}
    if not path.is_dir():
        return result
    # Match guide discovery: package + pipelines, or either with diffsynth metadata.
    for relative in ("diffsynth/__init__.py", "diffsynth/pipelines"):
        if ((path/relative).is_file() if relative.endswith(".py") else (path/relative).is_dir()):
            result["markers"].append(relative)
    for relative in ("pyproject.toml", "setup.py", "setup.cfg"):
        try:
            text = (path/relative).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if re.search(r'''\bname\s*=\s*(["'])diffsynth\1''', text) or re.search(r"(?m)^\s*name\s*=\s*diffsynth\s*$", text):
            result["markers"].append(relative)
    packages = {"diffsynth/__init__.py", "diffsynth/pipelines"}.intersection(result["markers"])
    recognized = len(packages) == 2 or (bool(packages) and len(result["markers"]) > len(packages))
    result["status"] = ("ok" if len(packages) == 2 else "partial") if recognized else "invalid_source_root"
    return result


def origin_match(origin, expected):
    if not origin or not expected:
        return "unknown"
    try:
        expected = Path(expected).resolve()
        if not expected.is_file():
            return "unknown"
        return "match" if Path(origin).resolve() == expected else "mismatch"
    except (OSError, ValueError, RuntimeError):
        return "unknown"


def python_requirement(root=None, current=None):
    """Read the active project's declared Python floor without importing it.

    Only evaluate a simple >=release constraint; leave other PEP 440 forms to
    pip rather than implementing a partial general-purpose dependency resolver.
    """
    result = {"status": "not_assessed", "requires_python": None, "source": None, "errors": []}
    try:
        if root:
            path = Path(root).expanduser().resolve()/'pyproject.toml'
            if not path.is_file():
                return result  # Exported sources may omit packaging metadata.
            text = path.read_text(encoding="utf-8")
            project = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", text)
            declaration = re.search(r'''(?m)^requires-python\s*=\s*(["'])([^"']+)\1''', project[1]) if project else None
            result.update(requires_python=declaration[2] if declaration else None, source=str(path))
        else:
            result.update(requires_python=importlib.metadata.metadata("diffsynth").get("Requires-Python"),
                          source="installed_distribution")
        if result["requires_python"]:
            floor = re.fullmatch(r"\s*>=\s*(\d+)\.(\d+)(?:\.(\d+))?\s*", result["requires_python"])
            if floor:
                minimum = tuple(int(part or 0) for part in floor.groups())
                current = sys.version_info if current is None else current
                compatible = tuple(current[:3]) >= minimum
                if tuple(current[:3]) == minimum and len(current) > 3 and current[3] != "final":
                    compatible = False
                result["status"] = "pass" if compatible else "fail"
    except importlib.metadata.PackageNotFoundError:
        pass
    except (OSError, UnicodeError, ValueError) as error:
        result["errors"].append(issue("framework_python_requirement", error))
    return result


def probe(roots, initialize_torch=False, model_dir=None, output_dir=None, import_diffsynth=False, timeout=30,
          reserve_gib=0.5, weight_files=()):
    if set(roots)-{"studio"}:
        raise ValueError("Only the Studio source root is supported by this helper")
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("reserve_gib must be finite and nonnegative")
    data = {"schema_version": 2, "observed_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.executable, "python_version": platform.python_version(),
            "platform": platform.platform(), "packages": {}, "sources": {}, "storage": {}, "warnings": []}
    data.update(torch_state(initialize_torch))
    for name in ("diffsynth", "torch", "torchvision", "transformers", "accelerate", "deepspeed", "peft", "safetensors",
                 "einops", "imageio", "modelscope", "ftfy", "pandas", "bitsandbytes", "torchao", "entropack", "comfy-kitchen", "deep-gemm", "triton", "Pillow"):
        try:
            data["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            data["packages"][name] = None
        except Exception as error:
            data["packages"][name] = None
            data["errors"].append(issue("package_metadata", error, package=name))
    root = roots.get("studio")
    data["scope"] = "python_and_source" if root else "python_environment_only"
    try:
        data["source_validation"] = source_validation(root)
    except (OSError, ValueError, RuntimeError) as error:
        data["source_validation"] = {"status": "source_invalid", "path": str(root), "markers": []}
        data["errors"].append(issue("source_validation", error))
    source_status = data["source_validation"]["status"]
    if source_status in ("source_invalid", "invalid_source_root", "partial"):
        data["errors"].append(issue("source_validation", ValueError("Requested source root is invalid or incomplete"), code=source_status))
    if root:
        path = Path(data["source_validation"]["path"])
        data["sources"]["studio"] = {"path": str(path), "exists": path.is_dir(), "git": git_info(path)}
    data["python_requirement"] = python_requirement(root) if source_status in ("ok", "partial", "not_requested") else {
        "status": "not_assessed", "requires_python": None, "source": None, "errors": []}
    data["errors"].extend(data["python_requirement"]["errors"])
    if data["python_requirement"]["status"] == "fail":
        data["errors"].append(issue("framework_python_requirement", ValueError(
            "Current Python does not satisfy "+data["python_requirement"]["requires_python"])))
    metadata = metadata_info(data["packages"]["diffsynth"], root)
    data["metadata_discovery"] = metadata
    data["errors"].extend(metadata["errors"])
    for role, target in (("cwd", Path.cwd()),
                         ("model", model_dir if model_dir is not None else os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")),
                         ("output", output_dir)):
        try:
            data["storage"][role] = storage_info(target)
        except (OSError, ValueError, RuntimeError) as error:
            data["storage"][role] = {"status": "invalid_path" if isinstance(error, ValueError) else "failed", "path": str(target)}
            data["errors"].append(issue("storage", error, role=role, path=str(target)))
    data["ram"] = memory_info()
    data["errors"].extend(data["ram"]["errors"])
    data["diffsynth_env"] = {key: os.environ[key] for key in ("DIFFSYNTH_MODEL_BASE_PATH", "DIFFSYNTH_DOWNLOAD_SOURCE", "DIFFSYNTH_SKIP_DOWNLOAD", "DIFFSYNTH_ATTENTION_IMPLEMENTATION", "DIFFSYNTH_FLASH_ATTN_KERNEL_REPO_ID", "DIFFSYNTH_FLASH_ATTN_KERNEL_VERSION", "DIFFSYNTH_DISK_MAP_BUFFER_SIZE") if key in os.environ}
    data["device_visibility"] = {key: os.environ[key] for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK") if key in os.environ}
    data["import_test"] = {"status": "not_tested", "origin": None,
                           "requested_source_match": "not_tested", "distribution_origin_match": "not_tested",
                           "requested_source_import_verified": False}
    if import_diffsynth:
        observed = run_worker("probe_import", timeout)
        data["import_test"] = observed.get("result", {"status": observed["status"], "origin": None})
        metadata["import_verified"] = data["import_test"]["status"] == "pass"
        actual = data["import_test"]
        actual["worker_diagnostics"] = observed.get("diagnostics")
        actual["requested_source_match"] = (origin_match(actual.get("origin"), Path(data["source_validation"]["path"])/'diffsynth/__init__.py')
                                            if root else "not_requested")
        actual["distribution_origin_match"] = origin_match(actual.get("origin"), metadata.get("distribution_origin"))
        actual["requested_source_import_verified"] = (actual["status"] == "pass" and source_status in ("ok", "partial")
            and actual["requested_source_match"] == "match")
        if actual["requested_source_match"] == "mismatch":
            data["errors"].append(issue("diffsynth_import_source", ValueError("Imported package origin differs from requested_source")))
        if actual["distribution_origin_match"] == "mismatch":
            data["warnings"].append({"stage": "diffsynth_install_metadata", "code": "distribution_origin_mismatch",
                                     "message": "Imported source differs from installed package metadata; inspect the editable install before relying on its version or dependency metadata."})
        data["errors"].extend(observed["errors"]+data["import_test"].get("errors", []))
    if initialize_torch:
        observed = run_worker("probe_torch", timeout)
        data["torch_worker_diagnostics"] = observed.get("diagnostics")
        if observed["status"] == "ok":
            fields = observed["result"]
            data["errors"].extend(fields.pop("errors", []))
            data.update(fields)
        else:
            data["device_probe_status"] = observed["status"]
            data["errors"].extend(observed["errors"])
    for device in data["devices"]:
        if "free_bytes" in device and "total_bytes" in device:
            try:
                device["memory_budget"] = vram_budget(device["free_bytes"], device["total_bytes"], reserve_gib)
            except ValueError as error:
                device["memory_budget"] = {"status": "unavailable", "reserve_gib": reserve_gib,
                                           "new_task_budget_gib": None, "vram_limit_gib": None,
                                           "reason": str(error)}
                data["errors"].append(issue("vram_budget", error, device=device["device"]))
    # Keep file rows separate: each component may have its own dtype and converter.
    data["weight_files"] = [weight_file_info(path) for path in dict.fromkeys(
        str(Path(path).expanduser().resolve()) for path in weight_files)]
    for row in data["weight_files"]:
        if row["status"] != "header_read":
            data["errors"].append(issue("weight_header", ValueError(row["error"]), path=row["path"]))
    data["validation"] = {"python_requirement": data["python_requirement"]["status"],
                          "package_import": data["import_test"]["status"],
                          "device_query": data["device_probe_status"],
                          "tensor_op": "not_tested", "backend_op": "not_tested", "pipeline": "not_tested"}
    data["diagnostic_status"] = "partial" if data["errors"] else "collected"
    data["target_readiness"] = "not_assessed"
    return redact_report(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio", help="Optional source root; without it only the Python environment is checked")
    parser.add_argument("--model-dir", help="Absolute model/cache storage path; env DIFFSYNTH_MODEL_BASE_PATH is used only if also absolute")
    parser.add_argument("--output-dir", help="Absolute inference/training output path, independent of the report file")
    parser.add_argument("--torch", action="store_true", help="Import torch and query devices in a subprocess; may initialize runtime, performs no tensor operation")
    parser.add_argument("--import-diffsynth", action="store_true", help="Test framework import in a subprocess; may initialize dependencies, does not load a model")
    parser.add_argument("--timeout", type=float, default=30, help="Timeout in seconds for each opt-in subprocess (default: 30)")
    parser.add_argument("--reserve-gib", type=float, default=0.5, help="VRAM reserve in GiB for snapshot arithmetic; the default is an inference reference, not a measured training peak or capacity guarantee")
    parser.add_argument("--weight-file", action="append", default=[], help="Optional local safetensors file; repeat for the actual component shards. Reads headers only, does not download or select quantization")
    parser.add_argument("--output", type=Path, help="Diagnostic report path; prefer the user task directory outside the source checkout")
    args = parser.parse_args()
    if not 0 < args.timeout < float('inf'):
        parser.error("--timeout must be finite and positive")
    try:
        data = probe({"studio": args.studio}, args.torch, args.model_dir, args.output_dir, args.import_diffsynth, args.timeout,
                     reserve_gib=args.reserve_gib, weight_files=args.weight_file)
        rendered = json.dumps(data, ensure_ascii=False, indent=2)+"\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, ValueError) as error:
        parser.exit(2, str(error)+"\n")


if __name__ == "__main__":
    main()
