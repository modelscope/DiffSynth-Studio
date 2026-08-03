from __future__ import annotations

import os
from typing import Any, Dict

from . import config, db


SETTING_KEYS: Dict[str, Dict[str, Any]] = {
    "DASHSCOPE_API_KEY": {
        "env": "DASHSCOPE_API_KEY",
        "default": "",
        "label": "阿里云百炼 API Key",
        "secret": True,
        "training_env": False,
    },
    "DASHSCOPE_BASE_URL": {
        "env": "DASHSCOPE_BASE_URL",
        "default": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "label": "阿里云百炼 Base URL",
        "training_env": False,
    },
    "DIFFSYNTH_MODEL_BASE_PATH": {
        "env": "DIFFSYNTH_MODEL_BASE_PATH",
        "default": "",
        "label": "模型下载路径",
    },
    "DIFFSYNTH_DOWNLOAD_SOURCE": {
        "env": "DIFFSYNTH_DOWNLOAD_SOURCE",
        "default": "modelscope",
        "label": "模型下载源 (modelscope / huggingface)",
    },
    "DIFFSYNTH_ATTENTION_IMPLEMENTATION": {
        "env": "DIFFSYNTH_ATTENTION_IMPLEMENTATION",
        "default": "",
        "label": "Attention 实现 (flash_attn / sage_attn / sdpa / xformers)",
    },
    "MODEL_SAVE_ROOT": {
        "env": "DIFFSYNTH_OUTPUTS_ROOT",
        "default": "",
        "label": "模型保存路径 (训练产物根目录)",
    },
    "DATASETS_ROOT": {
        "env": "DIFFSYNTH_DATASETS_ROOT",
        "default": "",
        "label": "数据集根目录",
    },
}


def get_all() -> Dict[str, str]:
    with db.settings_conn() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    saved = {r["key"]: r["value"] for r in rows}
    out: Dict[str, str] = {}
    for k, meta in SETTING_KEYS.items():
        if k in saved:
            out[k] = saved[k]
        elif meta["env"] and os.environ.get(meta["env"]):
            out[k] = os.environ[meta["env"]]
        else:
            out[k] = meta["default"]
    return out


def get_public() -> Dict[str, Any]:
    values = get_all()
    secret_configured: Dict[str, bool] = {}
    for key, meta in SETTING_KEYS.items():
        if meta.get("secret"):
            secret_configured[key] = bool(values.get(key, "").strip())
            values[key] = ""
    return {"settings": values, "secret_configured": secret_configured}


def set_many(values: Dict[str, str]) -> None:
    for key in ("DATASETS_ROOT", "MODEL_SAVE_ROOT"):
        value = str(values.get(key, "")).strip()
        if value:
            config.resolve_ui_path(value).mkdir(parents=True, exist_ok=True)
    with db.settings_conn() as conn:
        for k, v in values.items():
            if k not in SETTING_KEYS:
                continue
            if SETTING_KEYS[k].get("secret") and not str(v).strip():
                continue
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (k, v or ""),
            )


def clear_secret(key: str) -> None:
    meta = SETTING_KEYS.get(key)
    if not meta or not meta.get("secret"):
        raise ValueError(f"不是可清除的密钥设置: {key}")
    with db.settings_conn() as conn:
        conn.execute("DELETE FROM settings WHERE key = ?", (key,))
    env_key = meta.get("env")
    if env_key:
        os.environ.pop(str(env_key), None)


def apply_path_settings() -> None:
    values = get_all()
    dataset_value = values.get("DATASETS_ROOT", "").strip()
    output_value = values.get("MODEL_SAVE_ROOT", "").strip()
    config.DATASETS_ROOT = (
        config.resolve_ui_path(dataset_value)
        if dataset_value
        else config.DEFAULT_DATASETS_ROOT
    )
    config.OUTPUTS_ROOT = (
        config.resolve_ui_path(output_value)
        if output_value
        else config.DEFAULT_OUTPUTS_ROOT
    )
    config.ensure_dirs()


def build_env(base_env: Dict[str, str] | None = None) -> Dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)
    all_values = get_all()
    for k, meta in SETTING_KEYS.items():
        env_key = meta["env"]
        if not env_key:
            continue
        if not meta.get("training_env", True):
            env.pop(env_key, None)
            continue
        v = all_values.get(k, "")
        if v:
            env[env_key] = v
    return env
