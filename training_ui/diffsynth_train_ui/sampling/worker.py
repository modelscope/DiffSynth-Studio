"""Prepare and supervise isolated post-training sample processes."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import config, jobs, recipes
from .compat import adapted_source
from .schema import SampleRequest, SamplingConfig


SAMPLING_CONFIG_FILE = "sampling_config.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _prompts(config_data: Dict[str, Any]) -> List[str]:
    values = config_data.get("sample_prompts") or []
    if isinstance(values, str):
        values = values.splitlines()
    return [str(value).strip() for value in values if str(value).strip()]


def _latest_checkpoint(output_path: str) -> Optional[Path]:
    root = Path(output_path)
    candidates = [path for path in root.rglob("*.safetensors") if path.is_file()]
    if not candidates:
        return None

    def order(path: Path) -> Tuple[int, int, float, str]:
        match = re.search(r"(epoch|step)[-_]?(\d+)", path.stem, re.I)
        kind = 1 if match and match.group(1).lower() == "epoch" else 0
        number = int(match.group(2)) if match else -1
        return kind, number, path.stat().st_mtime, str(path)

    return max(candidates, key=order)


def _fail(run_id: str, message: str, returncode: Optional[int] = None) -> None:
    jobs.update_run(
        run_id,
        sampling_status="failed",
        sampling_returncode=returncode,
        sampling_message=message,
        sampling_finished_at=_now(),
        sampling_pid=None,
    )


def _prepare_sampling_config(run: jobs.JobRunRecord) -> Optional[Path]:
    prompts = _prompts(run.config)
    if not prompts:
        jobs.update_run(
            run.id,
            sampling_status="skipped",
            sampling_message="未配置测试 prompt。",
            sampling_finished_at=_now(),
        )
        return None

    recipe = recipes.get_recipe(str(run.config["model_type"]))
    sampling = recipe.sampling
    script_value = str(sampling.get("validate_script") or "")
    extension = str(sampling.get("output_extension") or "")
    if not script_value or not extension.startswith("."):
        _fail(run.id, f"模型 {recipe.name} 缺少有效的 sampling 配置。")
        return None

    script = (config.DIFFSYNTH_STUDIO_ROOT / script_value).resolve()
    if not script.is_file():
        _fail(run.id, f"采样脚本不存在：{script_value}")
        return None
    checkpoint = _latest_checkpoint(run.output_path)
    if not checkpoint:
        _fail(run.id, "训练结束后没有找到 .safetensors checkpoint。")
        return None

    sample_dir = (Path(run.output_path) / "final_samples").resolve()
    sample_dir.mkdir(parents=True, exist_ok=True)
    plan = SamplingConfig(
        run_id=run.id,
        model_type=recipe.name,
        validate_script=str(script),
        checkpoint=str(checkpoint.resolve()),
        output_dir=str(sample_dir),
        samples=[
            SampleRequest(
                prompt=prompt,
                output=str(sample_dir / f"sample_{index:03d}{extension}"),
            )
            for index, prompt in enumerate(prompts, 1)
        ],
    )
    config_path = sample_dir / SAMPLING_CONFIG_FILE
    plan.write(config_path)
    return config_path.resolve()


def _run_isolated_sample(config_path: Path, sample_index: int) -> int:
    plan = SamplingConfig.load(config_path)
    if sample_index < 0 or sample_index >= len(plan.samples):
        raise IndexError(f"sample index out of range: {sample_index}")
    sample = plan.samples[sample_index]
    script = Path(plan.validate_script).resolve()
    checkpoint = Path(plan.checkpoint).resolve()
    output = Path(sample.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = adapted_source(script, checkpoint, sample.prompt, output)
    namespace = {"__file__": str(script), "__name__": "__main__"}
    exec(compile(source, str(script), "exec"), namespace)
    if not output.is_file():
        raise RuntimeError(f"validation script did not create {output}")
    return 0


def run_sampling(run_id: str) -> str:
    run = jobs.get_run(run_id)
    config_path = _prepare_sampling_config(run)
    if config_path is None:
        return jobs.get_run(run_id).sampling_status
    plan = SamplingConfig.load(config_path)
    log_path = Path(plan.output_dir) / "sampling.log"
    jobs.update_run(
        run_id,
        sampling_status="running",
        sampling_pid=None,
        sampling_returncode=None,
        sampling_current=0,
        sampling_total=len(plan.samples),
        sampling_checkpoint=plan.checkpoint,
        sampling_script=plan.validate_script,
        sampling_message="",
        sampling_started_at=_now(),
        sampling_finished_at=None,
    )

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for sample_index, sample in enumerate(plan.samples):
            display_index = sample_index + 1
            jobs.update_run(run_id, sampling_current=display_index, sampling_pid=None)
            log.write(
                f"\n===== Prompt {display_index} / {len(plan.samples)} =====\n"
                f"{sample.prompt}\n"
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "diffsynth_train_ui.sampling.worker",
                    "--config",
                    str(config_path),
                    "--sample-index",
                    str(sample_index),
                ],
                cwd=str(config.DIFFSYNTH_STUDIO_ROOT),
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            jobs.update_run(run_id, sampling_pid=process.pid)
            returncode = process.wait()
            if returncode != 0:
                _fail(
                    run_id,
                    f"第 {display_index} 个 prompt 采样失败，请查看 sampling.log。",
                    returncode,
                )
                return "failed"

    jobs.update_run(
        run_id,
        sampling_status="finished",
        sampling_pid=None,
        sampling_returncode=0,
        sampling_current=len(plan.samples),
        sampling_message="",
        sampling_finished_at=_now(),
    )
    return "finished"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    args = parser.parse_args()
    return _run_isolated_sample(args.config.resolve(), args.sample_index)


if __name__ == "__main__":
    raise SystemExit(main())
