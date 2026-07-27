"""Two-stage prompt rewriter for LingBot-Video (example-side helper).

The LingBot-Video DiT is trained on **structured JSON captions**, not free-form prose.
:func:`rewrite_prompt` turns a brief idea into that structured caption via a faithful
port of the original two-stage rewriter (``rewriter/rewriter_core.py`` +
``rewriter/inference.py``): stage 1 *expands* the idea into a natural-language caption
(base model, no LoRA), stage 2 *maps* it into the structured JSON (base model + a
stage-2 LoRA adapter).

This is a separate VLM + LoRA adapter, NOT the DiT — it is not shipped or downloaded
with the pipeline, which is why it lives here with the examples rather than in the
diffsynth core (the core keeps only ``normalize_caption``). The bundled
:class:`TransformersBackend` loads the rewriter VLM locally; :func:`make_backend` also
accepts any object exposing ``generate(text, image, use_lora) -> str`` so the rewriter
can be driven by a hosted / OpenAI-compatible endpoint without shipping the weights.

Typical use::

    from prompt_rewriter import rewrite_prompt          # sibling module in this dir
    caption = rewrite_prompt("a puppy running across a meadow", mode="t2v", duration=5)
    video = pipe(prompt=caption, ...)                    # caption is the structured JSON string
"""

import io
import json
import os
import re

# Optional deps: only needed for the local rewriter backend / image loading.
try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

from system_prompts import (
    VIDEO_STEP1_EXPAND, VIDEO_STEP2_MAP, IMAGE_STEP1_EXPAND, IMAGE_STEP2_MAP,
)
from diffsynth.pipelines.lingbot_video import normalize_caption


# mode -> (step1 system prompt, step2 system prompt, feed image?, add duration?)
MODES = {
    "t2v":  dict(s1=VIDEO_STEP1_EXPAND, s2=VIDEO_STEP2_MAP, image=False, duration=True),
    "ti2v": dict(s1=VIDEO_STEP1_EXPAND, s2=VIDEO_STEP2_MAP, image=True,  duration=True),
    "t2i":  dict(s1=IMAGE_STEP1_EXPAND, s2=IMAGE_STEP2_MAP, image=False, duration=False),
}


def _has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


def load_image(src):
    """Load a first-frame image from a local path / http(s) URL / PIL.Image -> RGB PIL."""
    if Image is None:
        raise ImportError("loading a first-frame image requires the Pillow package.")
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, str) and re.match(r"^https?://", src):
        if requests is None:
            raise ImportError("fetching an image URL requires the requests package.")
        return Image.open(io.BytesIO(requests.get(src, timeout=30).content)).convert("RGB")
    return Image.open(src).convert("RGB")


def _step1_text(mode, prompt, dur):
    sys = MODES[mode]["s1"]
    if mode == "t2i":
        return sys + "\n\nUser image prompt:\n" + prompt
    dur_line = f"\n\n视频时长：{dur} 秒" if _has_cjk(prompt) else f"\n\nVideo Duration: {dur} seconds"
    return sys + "\n\n" + prompt + dur_line


def _step2_text(mode, detailed, dur):
    sys = MODES[mode]["s2"]
    if mode == "t2i":
        return sys + "\n\nDETAILED CAPTION:\n" + detailed
    return (sys + f"\n\nVideo Duration: {dur} seconds\n\nDETAILED CAPTION:\n"
            + detailed + "\n\nOutput the JSON now.")


def parse_json(raw):
    """Parse the stage-2 output into a dict, or ``None`` if it cannot be parsed.

    VLMs occasionally emit unstable JSON (missing quotes, trailing commas, ``` fences),
    so we strip any code fence, then try the stdlib parser first and fall back to
    ``json_repair`` when it is installed (recommended for messy outputs)."""
    s = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.DOTALL)
    if m:
        s = m.group(1)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    if repair_json is not None:
        try:
            obj = repair_json(s, return_objects=True)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def save_caption(result, duration, path):
    """Save as ``{"caption": <structured JSON caption>, "duration": <seconds|null>}`` —
    exactly the ``prompt.json`` the pipeline / runner consume. ``duration`` is integer
    seconds for T2V/TI2V and ``None`` for T2I (a still image has no duration)."""
    dur = int(round(duration)) if MODES[result["mode"]]["duration"] else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"caption": result["json"], "duration": dur}, f, ensure_ascii=False, indent=2)
    return path


class TransformersBackend:
    """Local rewriter VLM + LoRA adapter (peft). stage1 = base (adapter disabled),
    stage2 = base + LoRA. Loads the rewriter model into memory; the weights are NOT
    the DiT — set ``base``/``adapter`` (or ``REWRITER_BASE_MODEL``/``REWRITER_ADAPTER``)
    to the rewriter VLM and its stage-2 adapter."""

    def __init__(self, base=None, adapter=None, device="auto", max_new_tokens=6144):
        import contextlib
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._contextlib = contextlib
        self._torch = torch

        base = base or os.environ.get("REWRITER_BASE_MODEL", "")
        adapter = adapter or os.environ.get("REWRITER_ADAPTER", "")
        if not base or not adapter:
            raise ValueError(
                "Set the rewriter base and adapter paths via base=/adapter= or the "
                "REWRITER_BASE_MODEL / REWRITER_ADAPTER environment variables."
            )
        self.processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(model, adapter).eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, text, image, use_lora):
        torch = self._torch
        content = ([{"type": "image", "image": image}] if image is not None else []) \
            + [{"type": "text", "text": text}]
        messages = [{"role": "user", "content": content}]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(
            text=[chat], images=([image] if image is not None else None), return_tensors="pt"
        ).to(self.model.device)
        # stage1 (expand): disable LoRA; stage2 (map): keep LoRA active.
        adapter_ctx = self._contextlib.nullcontext() if use_lora else self.model.disable_adapter()
        with torch.no_grad(), adapter_ctx:
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]


def make_backend(backend="transformers", base=None, adapter=None):
    """Build a rewriter backend.

    - ``"transformers"`` — the bundled local :class:`TransformersBackend`.
    - a custom object / callable — returned as-is if it already exposes
      ``generate(text, image, use_lora) -> str``. Use this to drive the rewriter from
      a hosted or OpenAI-compatible endpoint without downloading the VLM locally.
    """
    if backend == "transformers":
        return TransformersBackend(base, adapter)
    if hasattr(backend, "generate"):
        return backend
    raise ValueError(
        f"unknown backend: {backend!r}; pass 'transformers' or an object exposing "
        "generate(text, image, use_lora)."
    )


class Rewriter:
    """Two-stage orchestrator. The backend implements ``generate(text, image, use_lora) -> str``."""

    def __init__(self, backend):
        self.backend = backend

    def rewrite(self, prompt, mode="t2v", first_frame=None, duration=5):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        cfg = MODES[mode]
        dur = int(round(duration))
        img = None
        if cfg["image"]:
            if first_frame is None:
                raise ValueError(f"{mode} requires first_frame (path / URL / PIL.Image)")
            img = load_image(first_frame)
        # stage 1: EXPAND -- base model (no LoRA)
        detailed = self.backend.generate(_step1_text(mode, prompt, dur), img, use_lora=False).strip()
        # stage 2: MAP -- base + LoRA
        raw = self.backend.generate(_step2_text(mode, detailed, dur), img, use_lora=True).strip()
        return {"mode": mode, "detailed": detailed, "json": parse_json(raw), "json_raw": raw}


def rewrite_prompt(prompt, mode="t2v", first_frame=None, duration=5,
                   backend="transformers", base=None, adapter=None, return_result=False):
    """Rewrite a brief idea into the structured caption string the pipeline expects.

    Returns the compact-JSON caption string (ready to pass as ``pipe(prompt=...)``).
    Pass ``return_result=True`` to also get the full stage-1/stage-2 dict.
    """
    rw = Rewriter(make_backend(backend, base, adapter))
    result = rw.rewrite(prompt, mode=mode, first_frame=first_frame, duration=duration)
    caption = normalize_caption(result["json"]) if result["json"] is not None else result["json_raw"]
    if return_result:
        return caption, result
    return caption
