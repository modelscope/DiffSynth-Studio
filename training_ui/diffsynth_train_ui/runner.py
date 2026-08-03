from __future__ import annotations

import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config, datasets, jobs, recipes, settings


_TRAIN_EXIT_CODE_FILE = ".train_exit_code"
_TRAINING_CONFIG_FILE = "training_config.json"


def _subprocess_env() -> Dict[str, str]:
    env = settings.build_env()
    project_root = str(config.DIFFSYNTH_STUDIO_ROOT.resolve())
    training_ui_root = str(config.TRAINING_UI_ROOT.resolve())
    existing = env.get("PYTHONPATH", "")
    entries = [item for item in existing.split(os.pathsep) if item]
    preferred = {project_root, training_ui_root}
    entries = [item for item in entries if str(Path(item).resolve()) not in preferred]
    env["PYTHONPATH"] = os.pathsep.join([project_root, training_ui_root, *entries])
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _stringify_model_paths(model_paths: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    origin_parts: List[str] = []
    local_parts: List[str] = []
    fp8_parts: List[str] = []
    for mp in model_paths:
        model_id = (mp.get("model_id") or "").strip()
        file_pattern = (mp.get("file_pattern") or "").strip()
        local_path = (mp.get("local_path") or "").strip()
        fp8 = bool(mp.get("fp8"))
        if local_path:
            local_parts.append(local_path)
            if fp8:
                fp8_parts.append(local_path)
        elif model_id:
            spec = f"{model_id}:{file_pattern}" if file_pattern else model_id
            origin_parts.append(spec)
            if fp8:
                fp8_parts.append(spec)
    return (",".join(origin_parts), ",".join(local_parts), ",".join(fp8_parts))


def _resolve_extra_inputs(dataset_name: str) -> str:
    if not dataset_name:
        return ""
    try:
        keys = datasets.get_extra_input_keys(dataset_name)
    except Exception:
        return ""
    return ",".join(keys)


def _extra_input_arg(cfg: Dict[str, Any], dataset_name: str) -> str:
    explicit = (cfg.get("extra_inputs") or "").strip()
    if explicit:
        return explicit
    return _resolve_extra_inputs(dataset_name)


def build_command(
    cfg: Dict[str, Any],
    prepared_output_path: Optional[Path] = None,
) -> Tuple[List[str], str, str]:
    recipe = recipes.get_recipe(cfg["model_type"])
    ds_name = cfg.get("dataset") or ""
    ds_dir = datasets.dataset_path(ds_name)
    metadata_path = datasets.metadata_path(ds_name)

    origin, model_paths_str, fp8_str = _stringify_model_paths(cfg.get("model_paths") or [])

    if cfg.get("enable_custom_lora_target"):
        lora_target = (cfg.get("lora_target_modules") or "").strip()
    else:
        lora_target = recipe.default_lora_target

    optimizer = (cfg.get("optimizer") or recipe.default_optimizer).strip()
    remove_prefix = recipe.remove_prefix or recipes.remove_prefix_for(recipe.lora_base_model)

    if prepared_output_path is None:
        raise ValueError("build_command requires a run output directory")
    output_path = prepared_output_path
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "train.log"

    script = config.DIFFSYNTH_STUDIO_ROOT / recipe.train_script
    argv: List[str] = [sys.executable, str(script)]

    def add(k: str, v: Optional[Any]) -> None:
        if v is None:
            return
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
            return
        s = str(v)
        if s == "":
            return
        argv.extend([f"--{k}", s])

    add("dataset_base_path", str(ds_dir))
    add("dataset_metadata_path", str(metadata_path))
    add("dataset_repeat", cfg.get("dataset_repeat", recipe.default_dataset_repeat))
    if recipe.default_data_file_keys:
        add("data_file_keys", recipe.default_data_file_keys)

    if origin:
        add("model_id_with_origin_paths", origin)
    if model_paths_str:
        add("model_paths", model_paths_str)
    if fp8_str:
        add("fp8_models", fp8_str)

    add("lora_rank", cfg.get("lora_rank", recipe.default_lora_rank))
    argv.extend(["--lora_target_modules", lora_target])
    add("lora_base_model", recipe.lora_base_model)

    if "resolution" not in recipe.disable_sections:
        resolution_mode = cfg.get("resolution_mode") or recipe.default_resolution_mode
        if resolution_mode == "max_pixels":
            add("max_pixels", cfg.get("max_pixels", recipe.default_max_pixels))
        else:
            add("height", cfg.get("height", recipe.default_height))
            add("width", cfg.get("width", recipe.default_width))
    if recipe.dataset_kind == "video":
        add("num_frames", cfg.get("num_frames", recipe.default_num_frames))

    add("num_epochs", cfg.get("num_epochs", recipe.default_epochs))
    add("learning_rate", cfg.get("learning_rate", recipe.default_lr))
    add("customized_optimizer", optimizer)
    add("output_path", str(output_path))
    add("remove_prefix_in_ckpt", remove_prefix)
    if recipe.default_use_gradient_checkpointing:
        argv.append("--use_gradient_checkpointing")

    extra = _extra_input_arg(cfg, ds_name) or recipe.default_extra_inputs
    if extra:
        add("extra_inputs", extra)

    add("gradient_accumulation_steps", cfg.get("gradient_accumulation", recipe.default_gradient_accumulation))
    add("dataset_num_workers", cfg.get("dataset_num_workers", recipe.default_dataset_num_workers))
    if cfg.get("find_unused_parameters"):
        argv.append("--find_unused_parameters")

    for k, v in recipe.extra_defaults.items():
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
        elif v is not None and v != "":
            add(k, str(v))

    extra_cli = cfg.get("extra_cli") or []
    for tok in extra_cli:
        if tok:
            argv.append(str(tok))

    if len(recipe.training_stages) > 1:
        argv = _build_staged_command(argv, recipe, ds_dir, metadata_path, Path(output_path), cfg)

    return argv, str(output_path), str(log_path)


def prepare_run(run_id: str) -> jobs.JobRunRecord:
    run = jobs.get_run(run_id)
    cfg = dict(run.config)
    cfg.setdefault("job_name", run.job_name)
    recipe = recipes.get_recipe(cfg["model_type"])
    output_path = Path(run.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    argv, resolved_output, log_path = build_command(cfg, output_path)
    default_config_path = recipes.get_default_config_path(recipe.name)
    payload = {
        "job_id": run.job_id,
        "run_id": run.id,
        "job_name": run.job_name,
        "model_type": recipe.name,
        "model_default_config": str(default_config_path.relative_to(config.TRAINING_UI_ROOT)),
        "model_default_snapshot": json.loads(default_config_path.read_text(encoding="utf-8")),
        "user_config": cfg,
        "resolved": {
            "output_path": resolved_output,
            "log_path": log_path,
            "command": argv,
        },
        "prepared_at": _now(),
    }
    config_path = output_path / _TRAINING_CONFIG_FILE
    temp_path = output_path / f".{_TRAINING_CONFIG_FILE}.tmp"
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, config_path)
    jobs.update_run(
        run.id,
        command_json=json.dumps(argv, ensure_ascii=False),
        log_path=log_path,
    )
    return jobs.get_run(run.id)


def _read_prepared_training_config(run: jobs.JobRunRecord) -> Dict[str, Any]:
    if not run.output_path:
        raise FileNotFoundError("run has no output directory")
    path = Path(run.output_path) / _TRAINING_CONFIG_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("job_id") != run.job_id or payload.get("run_id") != run.id:
        raise ValueError(f"invalid prepared training config: {path}")
    resolved = payload.get("resolved") or {}
    command = resolved.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"training command missing in {path}")
    return payload


def _build_staged_command(
    base_argv: List[str],
    recipe: recipes.ModelRecipe,
    dataset_dir: Path,
    metadata_path: Path,
    output_path: Path,
    cfg: Dict[str, Any],
) -> List[str]:
    def remove_arg(command: List[str], key: str) -> None:
        flag = f"--{key}"
        while flag in command:
            index = command.index(flag)
            del command[index]
            if index < len(command) and not command[index].startswith("--"):
                del command[index]

    def set_arg(command: List[str], key: str, value: Any) -> None:
        remove_arg(command, key)
        if value is True:
            command.append(f"--{key}")
        elif value is not None and value != "":
            command.extend([f"--{key}", str(value)])

    commands: List[List[str]] = []
    previous_cache: Optional[Path] = None
    stage_specific = {
        "model_id_with_origin_paths",
        "model_paths",
        "fp8_models",
        "task",
        "max_timestep_boundary",
        "min_timestep_boundary",
        "processor_config",
        "noise_scale",
    }
    for index, stage in enumerate(recipe.training_stages):
        stage = dict(stage)
        raw_configured_stages = cfg.get("stage_parameters") or []
        configured_stages = raw_configured_stages if isinstance(raw_configured_stages, list) else []
        configured_stage = (
            configured_stages[index]
            if index < len(configured_stages) and isinstance(configured_stages[index], dict)
            else {}
        )
        for key in ("max_timestep_boundary", "min_timestep_boundary"):
            if key in stage and configured_stage.get(key) is not None:
                value = float(configured_stage[key])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"第 {index + 1} 阶段的 {key} 必须在 0 到 1 之间")
                stage[key] = str(value)
        minimum = stage.get("min_timestep_boundary")
        maximum = stage.get("max_timestep_boundary")
        if minimum is not None and maximum is not None and float(minimum) > float(maximum):
            raise ValueError(f"第 {index + 1} 阶段的 min_timestep_boundary 不能大于 max_timestep_boundary")
        command = list(base_argv)
        for key in stage_specific:
            remove_arg(command, key)
            if key in stage:
                set_arg(command, key, stage[key])

        uses_cache = previous_cache is not None
        set_arg(command, "dataset_base_path", previous_cache if uses_cache else dataset_dir)
        if uses_cache:
            remove_arg(command, "dataset_metadata_path")
        else:
            set_arg(command, "dataset_metadata_path", metadata_path)

        is_last = index == len(recipe.training_stages) - 1
        stage_output = output_path if is_last else output_path / f"stage_{index + 1}"
        set_arg(command, "output_path", stage_output)
        if recipe.dataset_repeat_stage_index == index:
            set_arg(command, "dataset_repeat", cfg.get("dataset_repeat", recipe.default_dataset_repeat))
        elif "dataset_repeat" in stage:
            set_arg(command, "dataset_repeat", stage["dataset_repeat"])
        else:
            set_arg(command, "dataset_repeat", cfg.get("dataset_repeat", recipe.default_dataset_repeat))

        commands.append(command)
        task = str(stage.get("task", ""))
        previous_cache = stage_output if task.endswith(":data_process") else None

    return ["bash", "-lc", " && ".join(shlex.join(command) for command in commands)]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _status_from_returncode(returncode: Optional[int]) -> str:
    if returncode is None:
        return "unknown"
    return "finished" if returncode == 0 else "failed"


def start_job(job_id: str) -> jobs.JobRecord:
    job = jobs.get_job(job_id)
    if job.latest_run and job.latest_run.status in jobs.ACTIVE_RUN_STATUSES:
        return job
    run = jobs.create_run(job_id)
    try:
        run = prepare_run(run.id)
        payload = _read_prepared_training_config(run)
    except Exception:
        jobs.update_run(run.id, status="failed", finished_at=_now())
        raise
    cfg = dict(payload["user_config"])
    resolved = payload["resolved"]
    argv = [str(item) for item in resolved["command"]]
    output_path = str(resolved["output_path"])
    log_path = str(resolved["log_path"])

    env = _subprocess_env()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu_index", 0))

    Path(output_path).mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "a", buffering=1, encoding="utf-8")
    logf.write(f"\n===== [{_now()}] 启动训练 =====\n")
    logf.flush()

    try:
        (Path(output_path) / _TRAIN_EXIT_CODE_FILE).unlink(missing_ok=True)
        proc = subprocess.Popen(
            [sys.executable, "-m", "diffsynth_train_ui.run_worker", "--run-id", run.id],
            cwd=str(config.DIFFSYNTH_STUDIO_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        logf.close()
        jobs.update_run(run.id, status="failed", finished_at=_now())
        raise
    logf.close()

    jobs.update_run(
        run.id,
        status="running",
        os_pid=proc.pid,
        started_at=_now(),
    )
    return jobs.get_job(job_id)


def stop_job(job_id: str) -> None:
    job = jobs.get_job(job_id)
    run = job.latest_run
    if not run:
        return
    if not run.os_pid:
        jobs.update_run(run.id, status="stopped", finished_at=_now())
        return
    try:
        os.killpg(os.getpgid(run.os_pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(10):
        if not is_alive(run.os_pid):
            break
        time.sleep(0.5)
    else:
        try:
            os.killpg(os.getpgid(run.os_pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    fields: Dict[str, Any] = {"status": "stopped", "finished_at": _now()}
    if run.status == "sampling":
        fields.update({
            "sampling_status": "stopped",
            "sampling_pid": None,
            "sampling_message": "用户停止了采样。",
            "sampling_finished_at": _now(),
        })
    jobs.update_run(run.id, **fields)


def is_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_training_exit_code(run: jobs.JobRunRecord) -> Optional[int]:
    if not run.output_path:
        return None
    path = Path(run.output_path) / _TRAIN_EXIT_CODE_FILE
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def refresh_status(job_id: str) -> jobs.JobRecord:
    job = jobs.get_job(job_id)
    run = job.latest_run
    if not run or run.status not in jobs.ACTIVE_RUN_STATUSES:
        return job
    rc: Optional[int] = None
    exited = False
    if run.os_pid:
        try:
            waited_pid, status = os.waitpid(run.os_pid, os.WNOHANG)
            if waited_pid == run.os_pid:
                exited = True
                if os.WIFEXITED(status):
                    rc = os.WEXITSTATUS(status)
                elif os.WIFSIGNALED(status):
                    rc = 128 + os.WTERMSIG(status)
        except ChildProcessError:
            exited = not is_alive(run.os_pid)
    if not exited and is_alive(run.os_pid):
        return job
    if rc is None:
        rc = _read_training_exit_code(run)
    new_status = _status_from_returncode(rc)
    jobs.update_run(run.id, status=new_status, returncode=rc, finished_at=_now())
    if run.status == "sampling" and rc == 0:
        jobs.update_run(
            run.id,
            sampling_status="failed",
            sampling_pid=None,
            sampling_message="采样 supervisor 意外退出。",
            sampling_finished_at=_now(),
        )
    return jobs.get_job(job_id)


def read_log(job_id: str) -> str:
    job = jobs.get_job(job_id)
    run = job.latest_run
    if not run or not run.log_path or not Path(run.log_path).is_file():
        return "尚无日志"
    return Path(run.log_path).read_text(encoding="utf-8", errors="replace")
