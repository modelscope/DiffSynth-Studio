from __future__ import annotations

import argparse
import os
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

from . import config, jobs
from .sampling import run_sampling


def _write_exit_code(path: Path, returncode: int) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(f"{returncode}\n", encoding="utf-8")
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run = jobs.get_run(args.run_id)

    for _ in range(100):
        if jobs.get_run(run.id).status == "running":
            break
        time.sleep(0.02)

    process = subprocess.Popen(
        run.command,
        cwd=str(config.DIFFSYNTH_STUDIO_ROOT),
        env=os.environ.copy(),
    )
    returncode = process.wait()
    _write_exit_code(Path(run.output_path) / ".train_exit_code", returncode)
    if returncode != 0:
        jobs.update_run(
            run.id,
            status="failed",
            returncode=returncode,
            finished_at=_now(),
        )
        return returncode

    prompts = run.config.get("sample_prompts") or []
    if isinstance(prompts, str):
        prompts = prompts.splitlines()
    prompt_total = sum(1 for prompt in prompts if str(prompt).strip())
    jobs.update_run(
        run.id,
        status="sampling",
        returncode=0,
        sampling_status="queued" if prompt_total else "not_started",
        sampling_current=0,
        sampling_total=prompt_total,
    )
    try:
        run_sampling(run.id)
    except Exception as exc:
        sample_dir = Path(run.output_path) / "final_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        with (sample_dir / "sampling.log").open("a", encoding="utf-8") as log:
            log.write("\n===== Sampling worker error =====\n")
            traceback.print_exc(file=log)
        jobs.update_run(
            run.id,
            sampling_status="failed",
            sampling_pid=None,
            sampling_message=f"采样进程异常：{exc}",
            sampling_finished_at=_now(),
        )
    jobs.update_run(run.id, status="finished", returncode=0, finished_at=_now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
