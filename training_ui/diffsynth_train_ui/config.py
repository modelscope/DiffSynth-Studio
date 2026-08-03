from __future__ import annotations

import os
from pathlib import Path


TRAINING_UI_ROOT: Path = Path(__file__).resolve().parents[1]


def resolve_ui_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = TRAINING_UI_ROOT / path
    return path.resolve()


def _resolve_ds_root() -> Path:
    env = os.environ.get("DIFFSYNTH_STUDIO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    inferred = Path(__file__).resolve().parents[2]
    if (inferred / "examples").is_dir():
        return inferred

DIFFSYNTH_STUDIO_ROOT: Path = _resolve_ds_root()

UI_DATA_ROOT: Path = resolve_ui_path(
    os.environ.get("DIFFSYNTH_TRAIN_UI_HOME", str(TRAINING_UI_ROOT / "data"))
)

DEFAULT_DATASETS_ROOT: Path = UI_DATA_ROOT / "datasets"
DEFAULT_OUTPUTS_ROOT: Path = UI_DATA_ROOT / "outputs"
DATASETS_ROOT: Path = resolve_ui_path(
    os.environ.get("DIFFSYNTH_DATASETS_ROOT", str(DEFAULT_DATASETS_ROOT))
)

OUTPUTS_ROOT: Path = resolve_ui_path(
    os.environ.get("DIFFSYNTH_OUTPUTS_ROOT", str(DEFAULT_OUTPUTS_ROOT))
)

DB_PATH: Path = UI_DATA_ROOT / "jobs.sqlite"

SETTINGS_DB_PATH: Path = UI_DATA_ROOT / "settings.sqlite"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
COMPRESSED_EXTS = {".zip", ".tar", ".tar.gz", ".tgz"}


def ensure_dirs() -> None:
    for p in (UI_DATA_ROOT, DATASETS_ROOT, OUTPUTS_ROOT):
        p.mkdir(parents=True, exist_ok=True)
