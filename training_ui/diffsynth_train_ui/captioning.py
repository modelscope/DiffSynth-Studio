from __future__ import annotations

import base64
import http.client
import io
import json
import socket
import ssl
from pathlib import Path
from typing import Any, Dict
from urllib import error, request
from urllib.parse import urlparse

from PIL import Image

from . import settings


DEFAULT_INSTRUCTION = "请生成准确、自然、适合图像生成模型训练的英文 prompt。只返回 prompt 正文。"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class CaptioningConfigurationError(ValueError):
    """Raised when Model Studio settings are incomplete or invalid."""


def _image_data_url(path: Path) -> str:
    with Image.open(path) as image:
        image.thumbnail((2048, 2048))
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")

        encoded = io.BytesIO()
        quality = 90
        while True:
            encoded.seek(0)
            encoded.truncate(0)
            image.save(encoded, format="JPEG", quality=quality, optimize=True)
            if encoded.tell() <= 6 * 1024 * 1024 or quality <= 55:
                break
            quality -= 10
    data = base64.b64encode(encoded.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _validate_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return base_url


def _response_text(payload: Dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("响应中没有可用的 prompt") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(parts).strip()
    raise RuntimeError("返回了无法识别的内容格式")


def generate_prompt(
    image_path: Path,
    model: str,
    current_prompt: str = "",
    instruction: str = "",
) -> str:
    values = settings.get_all()
    api_key = values.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise CaptioningConfigurationError("尚未配置 API Key，请先到设置页面配置")
    base_url = _validate_base_url(values.get("DASHSCOPE_BASE_URL", ""))
    model = model.strip()
    if not model:
        raise CaptioningConfigurationError("请选择用于生成 Prompt 的模型")

    user_instruction = (instruction or DEFAULT_INSTRUCTION).strip()[:4000]
    existing = current_prompt.strip()[:12000]
    text = user_instruction
    if existing:
        text += f"\n\n当前 prompt：\n{existing}\n\n请结合图像修改并改进当前 prompt。"
    text += "\n不要解释，不要使用 Markdown，不要添加引号。"

    body = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                {"type": "text", "text": text},
            ],
        }],
        "temperature": 0.2,
        "max_tokens": 512,
        "enable_thinking": False,
    }).encode("utf-8")
    req = request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            raw = response.read(_MAX_RESPONSE_BYTES)
    except error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(f"请求失败 ({exc.code}): {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"无法连接服务: {exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError("请求超时，请稍后重试或更换响应更快的模型") from exc
    except (ConnectionError, http.client.RemoteDisconnected, ssl.SSLError, OSError) as exc:
        raise RuntimeError(f"连接中断: {exc}") from exc

    try:
        result = _response_text(json.loads(raw.decode("utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("返回了无效 JSON") from exc
    if not result:
        raise RuntimeError("返回了空 prompt")
    return result
