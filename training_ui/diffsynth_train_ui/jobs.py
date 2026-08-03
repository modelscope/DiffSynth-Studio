from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config, db, settings


ACTIVE_RUN_STATUSES = ("preparing", "running", "sampling")


@dataclass
class JobRunRecord:
    id: str
    job_id: str
    job_name: str
    status: str
    config_json: str
    command_json: str = ""
    output_path: str = ""
    log_path: str = ""
    os_pid: Optional[int] = None
    returncode: Optional[int] = None
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    sampling_status: str = "not_started"
    sampling_pid: Optional[int] = None
    sampling_returncode: Optional[int] = None
    sampling_current: int = 0
    sampling_total: int = 0
    sampling_checkpoint: str = ""
    sampling_script: str = ""
    sampling_message: str = ""
    sampling_started_at: Optional[str] = None
    sampling_finished_at: Optional[str] = None

    @property
    def config(self) -> Dict[str, Any]:
        return json.loads(self.config_json) if self.config_json else {}

    @property
    def command(self) -> List[str]:
        return json.loads(self.command_json) if self.command_json else []

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["config"] = self.config
        result["command"] = self.command
        result["pid"] = self.os_pid
        return result


@dataclass
class JobRecord:
    id: str
    name: str
    model_type: str
    dataset: str
    config_json: str
    task_dir: str
    created_at: str
    updated_at: str
    latest_run: Optional[JobRunRecord] = None

    @property
    def config(self) -> Dict[str, Any]:
        return json.loads(self.config_json) if self.config_json else {}

    @property
    def status(self) -> str:
        return self.latest_run.status if self.latest_run else "created"

    @property
    def command(self) -> List[str]:
        return self.latest_run.command if self.latest_run else []

    @property
    def output_path(self) -> str:
        return self.latest_run.output_path if self.latest_run else ""

    @property
    def log_path(self) -> str:
        return self.latest_run.log_path if self.latest_run else ""

    @property
    def pid(self) -> Optional[int]:
        return self.latest_run.os_pid if self.latest_run else None

    @property
    def returncode(self) -> Optional[int]:
        return self.latest_run.returncode if self.latest_run else None

    @property
    def started_at(self) -> Optional[str]:
        return self.latest_run.started_at if self.latest_run else None

    @property
    def finished_at(self) -> Optional[str]:
        return self.latest_run.finished_at if self.latest_run else None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "model_type": self.model_type,
            "dataset": self.dataset,
            "config_json": self.config_json,
            "config": self.config,
            "command_json": self.latest_run.command_json if self.latest_run else "",
            "command": self.command,
            "output_path": self.output_path,
            "log_path": self.log_path,
            "pid": self.pid,
            "returncode": self.returncode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "task_dir": self.task_dir,
            "latest_run_id": self.latest_run.id if self.latest_run else None,
            "latest_run": self.latest_run.to_dict() if self.latest_run else None,
        }
        return result


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_task_name(name: str) -> str:
    normalized = "".join(
        char if (char.isalnum() or char in "-_") else "_" for char in name.strip()
    ).strip("_")
    return (normalized or "task")[:80]


def _row_to_run(row) -> JobRunRecord:
    return JobRunRecord(
        id=row["id"], job_id=row["job_id"], job_name=row["job_name"],
        status=row["status"], config_json=row["config_json"],
        command_json=row["command_json"] or "", output_path=row["output_path"] or "",
        log_path=row["log_path"] or "", os_pid=row["os_pid"],
        returncode=row["returncode"], created_at=row["created_at"],
        started_at=row["started_at"], finished_at=row["finished_at"],
        sampling_status=row["sampling_status"] or "not_started",
        sampling_pid=row["sampling_pid"],
        sampling_returncode=row["sampling_returncode"],
        sampling_current=row["sampling_current"] or 0,
        sampling_total=row["sampling_total"] or 0,
        sampling_checkpoint=row["sampling_checkpoint"] or "",
        sampling_script=row["sampling_script"] or "",
        sampling_message=row["sampling_message"] or "",
        sampling_started_at=row["sampling_started_at"],
        sampling_finished_at=row["sampling_finished_at"],
    )


def _latest_run_with_conn(conn, job_id: str) -> Optional[JobRunRecord]:
    row = conn.execute(
        "SELECT * FROM job_runs WHERE job_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (job_id,),
    ).fetchone()
    return _row_to_run(row) if row else None


def _row_to_job(conn, row) -> JobRecord:
    task_dir = row["task_dir"] or str(
        config.OUTPUTS_ROOT / f"{row['id']}_{_safe_task_name(row['name'])}"
    )
    return JobRecord(
        id=row["id"], name=row["name"], model_type=row["model_type"],
        dataset=row["dataset"] or "", config_json=row["config_json"],
        task_dir=task_dir, created_at=row["created_at"],
        updated_at=row["updated_at"] or row["created_at"],
        latest_run=_latest_run_with_conn(conn, row["id"]),
    )


def create_job(name: str, model_type: str, dataset: str, config_data: Dict[str, Any]) -> JobRecord:
    settings.apply_path_settings()
    job_id = uuid.uuid4().hex[:12]
    task_dir = config.OUTPUTS_ROOT / f"{job_id}_{_safe_task_name(name)}"
    task_dir.mkdir(parents=True, exist_ok=False)
    now = _now()
    try:
        with db.jobs_conn() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, name, model_type, dataset, config_json,
                    created_at, updated_at, task_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, name, model_type, dataset, json.dumps(config_data, ensure_ascii=False), now, now, str(task_dir)),
            )
    except Exception:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise
    return get_job(job_id)


def list_jobs(status: Optional[str] = None) -> List[JobRecord]:
    with db.jobs_conn() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC").fetchall()
        result = [_row_to_job(conn, row) for row in rows]
    return [job for job in result if not status or job.status == status]


def get_job(job_id: str) -> JobRecord:
    with db.jobs_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise KeyError(f"job not found: {job_id}")
        return _row_to_job(conn, row)


def update_job(job_id: str, **fields: Any) -> None:
    allowed = {"model_type", "dataset", "config_json", "updated_at", "task_dir"}
    if not fields or not set(fields).issubset(allowed):
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"run fields cannot be stored on jobs: {sorted(invalid)}")
        return
    columns = ", ".join(f"{key} = ?" for key in fields)
    with db.jobs_conn() as conn:
        conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", [*fields.values(), job_id])


def edit_job(job_id: str, name: str, model_type: str, dataset: str, config_data: Dict[str, Any]) -> JobRecord:
    job = get_job(job_id)
    if name != job.name:
        raise ValueError("任务名称创建后不能修改")
    if job.latest_run and job.latest_run.status in ACTIVE_RUN_STATUSES:
        raise ValueError("运行中的任务不能编辑，请先停止任务")
    update_job(
        job_id, model_type=model_type, dataset=dataset,
        config_json=json.dumps(config_data, ensure_ascii=False), updated_at=_now(),
    )
    return get_job(job_id)


def create_run(job_id: str) -> JobRunRecord:
    job = get_job(job_id)
    if job.latest_run and job.latest_run.status in ACTIVE_RUN_STATUSES:
        raise ValueError("任务已有正在准备或运行的进程")
    run_id = uuid.uuid4().hex[:16]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task_dir = Path(job.task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    output_path = task_dir / timestamp
    output_path.mkdir(parents=False, exist_ok=False)
    now = _now()
    try:
        with db.jobs_conn() as conn:
            conn.execute(
                """
                INSERT INTO job_runs (
                    id, job_id, job_name, status, config_json, output_path, created_at
                ) VALUES (?, ?, ?, 'preparing', ?, ?, ?)
                """,
                (run_id, job.id, job.name, job.config_json, str(output_path), now),
            )
    except Exception:
        shutil.rmtree(output_path, ignore_errors=True)
        raise
    return get_run(run_id)


def get_run(run_id: str) -> JobRunRecord:
    with db.jobs_conn() as conn:
        row = conn.execute("SELECT * FROM job_runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        raise KeyError(f"run not found: {run_id}")
    return _row_to_run(row)


def latest_run(job_id: str) -> Optional[JobRunRecord]:
    with db.jobs_conn() as conn:
        return _latest_run_with_conn(conn, job_id)


def list_runs(job_id: str) -> List[JobRunRecord]:
    with db.jobs_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM job_runs WHERE job_id = ? ORDER BY created_at DESC, rowid DESC",
            (job_id,),
        ).fetchall()
    return [_row_to_run(row) for row in rows]


def update_run(run_id: str, **fields: Any) -> None:
    allowed = {
        "status", "command_json", "log_path", "os_pid", "returncode",
        "started_at", "finished_at",
        "sampling_status", "sampling_pid", "sampling_returncode",
        "sampling_current", "sampling_total", "sampling_checkpoint",
        "sampling_script", "sampling_message", "sampling_started_at",
        "sampling_finished_at",
    }
    invalid = set(fields) - allowed
    if invalid:
        raise ValueError(f"invalid run fields: {sorted(invalid)}")
    if not fields:
        return
    columns = ", ".join(f"{key} = ?" for key in fields)
    with db.jobs_conn() as conn:
        conn.execute(f"UPDATE job_runs SET {columns} WHERE id = ?", [*fields.values(), run_id])


def delete_job_records(job_id: str) -> None:
    with db.jobs_conn() as conn:
        conn.execute("DELETE FROM job_runs WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
