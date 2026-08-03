from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple

from . import config


_SCHEMA_JOBS = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    model_type    TEXT NOT NULL,
    dataset       TEXT,
    config_json   TEXT NOT NULL,
    task_dir      TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""

_SCHEMA_JOB_RUNS = """
CREATE TABLE IF NOT EXISTS job_runs (
    id            TEXT PRIMARY KEY,
    job_id        TEXT NOT NULL,
    job_name      TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'preparing',
    config_json   TEXT NOT NULL,
    command_json  TEXT,
    output_path   TEXT NOT NULL,
    log_path      TEXT,
    os_pid        INTEGER,
    returncode    INTEGER,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    sampling_status      TEXT NOT NULL DEFAULT 'not_started',
    sampling_pid         INTEGER,
    sampling_returncode  INTEGER,
    sampling_current     INTEGER NOT NULL DEFAULT 0,
    sampling_total       INTEGER NOT NULL DEFAULT 0,
    sampling_checkpoint  TEXT,
    sampling_script      TEXT,
    sampling_message     TEXT,
    sampling_started_at  TEXT,
    sampling_finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_job_runs_job_id ON job_runs (job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_runs_status ON job_runs (status);
"""

_CREATE_ACTIVE_RUN_INDEX = """
CREATE UNIQUE INDEX idx_job_runs_one_active
ON job_runs (job_id) WHERE status IN ('preparing', 'running', 'sampling')
"""

_SCHEMA_SETTINGS = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


_INIT_LOCK = threading.Lock()
_INITIALIZED_PATHS: Optional[Tuple[Path, Path]] = None


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _init_db(path: Path, schema: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(schema)


def _ensure_job_columns_and_migrate_runs(path: Path) -> None:
    """Split legacy execution fields out of jobs without losing its latest run."""
    with _connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA_JOB_RUNS)
        conn.execute("BEGIN IMMEDIATE")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        if "task_dir" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN task_dir TEXT")
        if "updated_at" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN updated_at TEXT")
        run_columns = {row[1] for row in conn.execute("PRAGMA table_info(job_runs)").fetchall()}
        sampling_columns = {
            "sampling_status": "TEXT NOT NULL DEFAULT 'not_started'",
            "sampling_pid": "INTEGER",
            "sampling_returncode": "INTEGER",
            "sampling_current": "INTEGER NOT NULL DEFAULT 0",
            "sampling_total": "INTEGER NOT NULL DEFAULT 0",
            "sampling_checkpoint": "TEXT",
            "sampling_script": "TEXT",
            "sampling_message": "TEXT",
            "sampling_started_at": "TEXT",
            "sampling_finished_at": "TEXT",
        }
        for column, declaration in sampling_columns.items():
            if column not in run_columns:
                conn.execute(f"ALTER TABLE job_runs ADD COLUMN {column} {declaration}")
        active_index = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            ("idx_job_runs_one_active",),
        ).fetchone()
        active_index_sql = str(active_index[0] or "") if active_index else ""
        if "sampling" not in active_index_sql:
            conn.execute("DROP INDEX IF EXISTS idx_job_runs_one_active")
            conn.execute(_CREATE_ACTIVE_RUN_INDEX)
        legacy_rows = (
            conn.execute("SELECT * FROM jobs WHERE COALESCE(output_path, '') != ''").fetchall()
            if "output_path" in columns else []
        )
        for row in legacy_rows:
            exists = conn.execute(
                "SELECT 1 FROM job_runs WHERE job_id = ? LIMIT 1", (row["id"],)
            ).fetchone()
            if exists:
                continue
            run_id = f"legacy_{row['id']}"
            conn.execute(
                """
                INSERT INTO job_runs (
                    id, job_id, job_name, status, config_json, command_json,
                    output_path, log_path, os_pid, returncode, created_at,
                    started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, row["id"], row["name"], row["status"], row["config_json"],
                    row["command_json"], row["output_path"], row["log_path"], row["pid"],
                    row["returncode"], row["started_at"] or row["created_at"],
                    row["started_at"], row["finished_at"],
                ),
            )
        legacy_columns = {"status", "command_json", "output_path", "log_path", "pid", "returncode", "started_at", "finished_at"}
        if columns & legacy_columns:
            conn.execute("ALTER TABLE jobs RENAME TO jobs_legacy_layout")
            conn.execute(_SCHEMA_JOBS.strip())
            conn.execute(
                """
                INSERT INTO jobs (id, name, model_type, dataset, config_json, task_dir, created_at, updated_at)
                SELECT id, name, model_type, dataset, config_json, task_dir, created_at,
                       COALESCE(updated_at, created_at)
                FROM jobs_legacy_layout
                """
            )
            conn.execute("DROP TABLE jobs_legacy_layout")
        else:
            conn.execute("UPDATE jobs SET updated_at = created_at WHERE updated_at IS NULL")


def init_all() -> None:
    global _INITIALIZED_PATHS
    paths = (config.DB_PATH.resolve(), config.SETTINGS_DB_PATH.resolve())
    if _INITIALIZED_PATHS == paths:
        return
    with _INIT_LOCK:
        if _INITIALIZED_PATHS == paths:
            return
        config.ensure_dirs()
        _init_db(paths[0], _SCHEMA_JOBS)
        _ensure_job_columns_and_migrate_runs(paths[0])
        _init_db(paths[1], _SCHEMA_SETTINGS)
        _INITIALIZED_PATHS = paths


@contextmanager
def jobs_conn() -> Iterator[sqlite3.Connection]:
    init_all()
    conn = _connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def settings_conn() -> Iterator[sqlite3.Connection]:
    init_all()
    conn = _connect(config.SETTINGS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
