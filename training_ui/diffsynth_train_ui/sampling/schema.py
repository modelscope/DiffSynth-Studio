from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class SampleRequest:
    prompt: str
    output: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleRequest":
        prompt = str(data.get("prompt") or "").strip()
        output = str(data.get("output") or "").strip()
        if not prompt:
            raise ValueError("sampling prompt cannot be empty")
        if not output:
            raise ValueError("sampling output cannot be empty")
        return cls(prompt=prompt, output=output)


@dataclass(frozen=True)
class SamplingConfig:
    run_id: str
    model_type: str
    validate_script: str
    checkpoint: str
    output_dir: str
    samples: List[SampleRequest]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingConfig":
        required = ("run_id", "model_type", "validate_script", "checkpoint", "output_dir")
        values = {key: str(data.get(key) or "").strip() for key in required}
        missing = [key for key, value in values.items() if not value]
        if missing:
            raise ValueError(f"sampling config missing fields: {', '.join(missing)}")
        raw_samples = data.get("samples")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError("sampling config must contain at least one sample")
        if not all(isinstance(item, dict) for item in raw_samples):
            raise ValueError("each sampling sample must be a JSON object")
        return cls(
            **values,
            samples=[SampleRequest.from_dict(item) for item in raw_samples],
        )

    @classmethod
    def load(cls, path: Path) -> "SamplingConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("sampling config root must be a JSON object")
        return cls.from_dict(data)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
