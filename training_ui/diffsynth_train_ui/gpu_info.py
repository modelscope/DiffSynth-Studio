from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, List


def _run(cmd: List[str], timeout: float = 3.0) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return ""


def get_gpus() -> List[Dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    query = "index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
    output = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    gpus: List[Dict[str, Any]] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": int(parts[2]),
                "memory_used_mb": int(parts[3]),
                "memory_free_mb": int(parts[4]),
                "utilization": int(parts[5]),
                "temperature": int(parts[6]),
            })
        except ValueError:
            continue
    return gpus
