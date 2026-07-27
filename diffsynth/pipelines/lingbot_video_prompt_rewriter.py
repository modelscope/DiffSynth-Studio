"""Caption normalization for LingBot-Video.

The LingBot-Video DiT is trained on **structured JSON captions**, not free-form
prose. Feeding a flat sentence is out-of-distribution and noticeably degrades
quality; feeding the structured caption the model expects restores it.

This module holds only the lightweight, dependency-free normalization the pipeline
and the training module need: :func:`normalize_caption` turns a caption expressed as
a ``dict`` / ``list`` / path to a ``prompt.json`` into the exact compact-JSON string
the DiT consumes (a plain string is passed through untouched). It mirrors the original
``lingbot_video.utils.caption_from_sample`` byte-for-byte.

Turning a *brief idea* into that structured caption is a separate, heavier step (a
two-stage VLM rewriter). That lives with the examples, out of the core, so importing
this module never drags in the rewriter's optional deps — see
``examples/lingbot_video/model_inference/prompt_rewriter.py``.
"""

import json
import os


# Keys that describe how to *render* the clip rather than its content. When a full
# sample dict is given without an explicit "caption" key, these are stripped before
# serialisation (kept identical to the original ``caption_from_sample``).
_RUNTIME_KEYS = {"duration", "fps", "height", "width", "num_frames", "resolution", "ratio"}


def _serialize_caption(caption) -> str:
    """dict/list -> compact JSON (the exact model format); anything else -> ``str()``."""
    if isinstance(caption, (dict, list)):
        return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
    return str(caption)


def _caption_from_sample(sample) -> str:
    """Port of ``lingbot_video.utils.caption_from_sample``.

    A *sample* dict either carries the structured caption under ``"caption"`` or IS
    the caption once the runtime keys are dropped.
    """
    if isinstance(sample, dict):
        if "caption" in sample:
            caption = sample["caption"]
        else:
            caption = {k: v for k, v in sample.items() if k not in _RUNTIME_KEYS}
    else:
        caption = sample
    return _serialize_caption(caption)


def normalize_caption(prompt):
    """Normalise a caption into the compact-JSON string the LingBot DiT expects.

    Accepts:

    - ``dict`` / ``list`` — a structured caption (or a full sample dict with a
      ``"caption"`` key), serialised via the original compact-JSON convention.
    - a path to a ``prompt.json`` file (``str`` ending in ``.json`` that exists) —
      loaded, then handled as the dict/list case.
    - any other ``str`` — returned unchanged (already a caption string, or free-form
      prose the caller intentionally wants to feed as-is).
    - ``None`` — returned unchanged.

    Plain strings are passed through, so this is safe to call unconditionally on
    prompts that are already in the right format.
    """
    if prompt is None:
        return prompt
    if isinstance(prompt, str):
        if prompt.endswith(".json") and os.path.isfile(prompt):
            with open(prompt, "r", encoding="utf-8") as f:
                prompt = json.load(f)
            return _caption_from_sample(prompt)
        return prompt
    if isinstance(prompt, (dict, list)):
        return _caption_from_sample(prompt)
    return str(prompt)
