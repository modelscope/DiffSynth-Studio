"""Per-segment captions for one long video, for segment-by-segment generation.

Handing every segment the same whole-clip caption tells each one to perform the
entire arc, so a clip with a one-way trajectory (a flower opening) snaps back to
the start of that arc at every seam. One caption per segment instead: segment k is
told what happens in segment k's own span. Measured on a 40s bloom with 56-frame
segments, this roughly halves the mean seam discontinuity at every overlap.

The whole clip is captioned first; each segment then gets its own local caption
with that global text as a naming reference only (including single-segment short
clips). Generation consumes the local caption. Handing a segment its
*predecessor's* caption instead collapses the output -- greedy decoding copies
a caption already nearly right for the shared frames.

Segments are given by the caller as half-open output-frame ranges
`segments = [(s1, t1), (s2, t2), ...]`, where segment k covers frames
[s_k, t_k): the same ranges the caller generates in, so there is exactly one
definition of the schedule. Segment lengths may differ and consecutive segments
may overlap.

Call it as a function: `caption_windows(video, output, segments)` captions every
segment and returns `{s: caption}`. The captioner is a 27B model and is released
when the call returns, but its GPU memory stays in PyTorch's cache, so run it in
a separate process if the generation pipeline that follows needs that memory
back.

The captioner weights are fetched through `ModelConfig`, so they land in the
framework's unified model directory (`./models` by default, overridable with
`DIFFSYNTH_MODEL_BASE_PATH`) instead of a machine-specific path.
"""
import json
import os

import numpy as np
import torch
from PIL import Image

from diffsynth.core import ModelConfig

CAPTIONER_MODEL_ID = "Qwen/Qwen3.8-27B"
NUM_FRAMES = 12          # frames sampled per caption; more frames, more tokens
IMG_SHORT = 560          # short side each frame is scaled to, to cap tokens
FPS = 24                 # output frame rate the g0 indices are expressed in
REASONING_EFFORT = "medium"
MAX_NEW_TOKENS = 2048    # thinking needs headroom beyond the caption itself

SYSTEM_PROMPT = """\
You are a precise visual captioning assistant for a video generation pipeline.
Write entirely in English. Output only valid JSON with the requested key.

Grounding rules:
- Treat the supplied images as visual evidence. Reference text is secondary
  and must never override the images.
- Describe the main visible subjects, their relevant appearance, the setting,
  and clearly supported actions or changes.
- Separate an observed state from a transition. An object being bent does not
  by itself show that it bends during this segment.
- Describe a transition only when earlier and later supplied frames support it.
  Do not invent its onset, completion, intermediate steps, or cause.
- Keep references to the same entity consistent. If a small structure cannot
  be identified reliably, describe its visible appearance rather than guessing
  its anatomical identity or treating it as a new object.
- Do not invent subtle movement to make a mostly static segment eventful.
  Conversely, do not suppress clearly visible changes.
- Do not claim that every frame or every detail is unchanged from sampled
  images alone.
- Prefer concrete visual description over interpretation or generic closing
  statements. Omit uncertain details that are not essential.
"""

GLOBAL_PROMPT = """\
The following {n} images are sampled across an entire video
approximately {dur} seconds long. They are presented in chronological order,
from the beginning to the end of the video.
They provide sparse coverage, not a complete record of every event.

Write a concise caption describing the video as a whole.

Requirements:
- Establish the main subjects, their arrangement, and the visible setting
  or background. Do not infer a specific location.
- Summarize the major visible changes in chronological order.
- When different subjects change at different times, preserve that distinction.
- Use consistent, visually grounded names for the subjects.
- Do not invent events between sampled frames or imply continuous observation.
- Do not claim a "complete process" or a "fully completed" transition unless
  the visible starting and ending states support that description.
- Describe appearance changes without guessing their causes.
- Do not describe shot type, camera movement, or lens focal length.
- Avoid exhaustive object inventories and generic closing sentences.
- Do not mention the supplied images or these instructions.

Prefer 60-120 words, but use fewer when appropriate. Maximum 150 words.

Return exactly:
{{
  "global_caption": "<caption>"
}}
"""

USER_PROMPT = """\
The following {n} images are sampled from one video segment
approximately {dur} seconds long. They are presented in chronological order,
from the beginning to the end of this segment.
They are samples, not every frame of the video.

Write a standalone caption of this segment only.

Before writing, check:
1. What is already true in the earliest supplied frame?
2. What visibly changes across the supplied frames?
3. What is true in the latest supplied frame?

Caption requirements:
- Briefly establish the main subjects, their arrangement, and the background.
- Prioritize the actions or changes that distinguish this segment.
- When a meaningful change is visible, describe it in temporal order.
  Include the starting or ending state when needed to make the change clear.
- If a process is already underway at the start, do not describe it as newly
  beginning. Use "continues" only when further change is visible here.
- If little changes, describe the visible state concisely. Do not invent
  swaying, shifting, unfolding, or other minor motion.
- Use "opens", "bends", "turns", "fades", or similar change verbs only when
  supported by differences between earlier and later frames.
- Describe visible color changes without inferring biological or physical causes.
- Do not describe shot type, camera movement, or lens focal length.
- Do not mention the supplied images, frame numbers, or these instructions.

Prefer 40-100 words, but use fewer for simple or mostly static segments.
Do not add details merely to reach a word count. Maximum 150 words.

Return exactly:
{{
  "local_caption": "<caption>"
}}
"""

POSITION_BLOCK = """\

Position of this segment in the video:
- Whole video length: approximately {total_dur} seconds.
- This segment covers approximately {start} s to {end} s.
- This is segment {index} of {count}.
"""

LOCAL_BLOCK = """\

Optional reference description of the whole video:
<global_reference>
{global_caption}
</global_reference>

Use this reference only to help identify and name subjects that are also
visible in the current segment.

Rules for using the reference:
- The current segment's images are the primary evidence.
- Reuse entity names only when the visual correspondence is clear.
- Do not inherit colors, poses, positions, openness, maturity, motion,
  or event timing from the reference.
- Do not retell an earlier or later event as if it happens in this segment.
- An event mentioned in the reference may already be complete, may still
  be underway, or may not occur in this segment at all.
- If the reference conflicts with the images, follow the images.
- If an identity or structure is uncertain, use a neutral visual description
  rather than repeating an unsupported specific label.
- Do not mention the reference, other segments, or these instructions.

Final check before returning JSON:
Every temporal claim must be supported by the current segment's images.
In particular, verify any claim that something starts, becomes, opens,
bends, turns, fades, or finishes.
"""


# ---------------------------------------------------------------------- frames
def read_all(path):
    """Decode the whole video once; windows are then sampled from memory.

    One decode beats re-seeking per window: there are typically dozens of
    overlapping windows to serve from the same file.
    """
    import imageio.v3 as iio
    frames = iio.imread(path, plugin="pyav", index=None)
    try:
        fps = float(iio.immeta(path, plugin="pyav")["fps"])
    except Exception:
        fps = float(FPS)
    return frames, fps


def available_frames(frames, src_fps):
    """How many output frames at FPS the source can fill.

    Output frame k reads source frame round(k*src_fps/FPS), so the ceiling is
    the number of k whose index still lands inside the file.
    """
    k = 0
    while int(round(k * src_fps / FPS)) < frames.shape[0]:
        k += 1
    return k


def _scale(img):
    w, h = img.size
    short = min(w, h)
    if short > IMG_SHORT:
        s = IMG_SHORT / short
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img


def segment_frames(frames, src_fps, start, end, n_sample, total):
    """Sample up to n_sample frames out of output-index range [start, end).

    Output frame k maps to source frame round(k*src_fps/FPS), the same mapping
    the generation loader uses; sampling the source uniformly instead would
    drift against the frames the model actually renders for this segment.

    The range is clamped to what the file covers, and if that leaves fewer than
    n_sample frames, fewer are sampled rather than duplicated. A schedule whose
    last segment overruns the end would otherwise clamp most positions onto the
    final frame and get described as static.

    Returns (images, span) where span is how many output frames they really
    cover -- what the prompt should be told the duration is.
    """
    T = frames.shape[0]
    last = min(end - 1, total - 1)
    if last < start:
        raise ValueError(
            f"segment [{start}, {end}) starts at or past the end of the video "
            f"({total} frames at {FPS}fps)")
    n = min(n_sample, last - start + 1)
    imgs = [_scale(Image.fromarray(frames[min(int(round(k * src_fps / FPS)), T - 1)]).convert("RGB"))
            for k in np.linspace(start, last, n, dtype=int)]
    return imgs, last - start + 1


# ------------------------------------------------------------------- inference
def resolve_captioner_path(model_path=None):
    """Local directory of the captioner, downloaded on first use.

    `model_path` is an optional local override; by default the weights are
    fetched from `CAPTIONER_MODEL_ID` through `ModelConfig`, which stores them
    under the framework's unified model directory.
    """
    if model_path is None:
        config = ModelConfig(model_id=CAPTIONER_MODEL_ID, origin_file_pattern="")
    else:
        config = ModelConfig(path=model_path)
    config.download_if_necessary()
    if config.path is None:
        raise FileNotFoundError(
            f"Captioner weights for {CAPTIONER_MODEL_ID} are missing; run without "
            "DIFFSYNTH_SKIP_DOWNLOAD to fetch them or pass model_path."
        )
    return config.path


def load_captioner(model_path=None, device="cuda"):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_path = resolve_captioner_path(model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True)
    model.eval()
    return model, processor


def _generate(model, processor, frames, text):
    """One forward pass; returns the decoded new tokens.

    Qwen3.8 is a thinking model: reasoning_effort goes through the chat template
    (it only accepts low/medium/xhigh) and the output is prefixed with
    <think>...</think>, which parse_output strips.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"} for _ in frames] + [{"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=False, reasoning_effort=REASONING_EFFORT)
    inputs = processor(text=prompt, images=frames, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    return processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def caption_global(model, processor, frames, duration):
    return _generate(model, processor, frames,
                     GLOBAL_PROMPT.format(n=len(frames), dur=f"{duration:.1f}"))


def caption_frames(model, processor, frames, duration, global_caption=None, position=None):
    """`global_caption` makes this segment use the whole clip's naming;
    `position` needs total_dur / start / end / index / count."""
    text = USER_PROMPT.format(n=len(frames), dur=f"{duration:.1f}")
    if position:
        text += POSITION_BLOCK.format(**position)
    if global_caption:
        text += LOCAL_BLOCK.format(global_caption=global_caption.strip())
    return _generate(model, processor, frames, text)


def parse_output(raw):
    """Pull the JSON out, tolerating a thinking block and surrounding prose."""
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    raw = raw.strip()
    for candidate in (raw, raw[raw.find("{"): raw.rfind("}") + 1] if "{" in raw and "}" in raw else ""):
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    return None


# ------------------------------------------------------------------ the driver
def check_segments(segments):
    """Validate `segments` and return them as sorted [(s, t)] half-open ranges.

    A later segment may start before the previous one ends -- those head frames
    are pinned to the previous segment's output -- but every start must be
    unique (captions are keyed by start frame) and every segment must still
    contain frames of its own.
    """
    spans = []
    for item in segments:
        try:
            s, t = (int(v) for v in item)
        except (TypeError, ValueError):
            raise ValueError(f"segment {item!r} is not a (start, end) pair of frame ids") from None
        if t <= s:
            raise ValueError(f"segment ({s}, {t}) covers no frames; segments are "
                             "half-open [start, end) output-frame ranges")
        spans.append((s, t))
    if not spans:
        raise ValueError("segments is empty")
    spans.sort()
    for (s0, t0), (s1, t1) in zip(spans, spans[1:]):
        if s1 == s0:
            raise ValueError(f"two segments start at frame {s0}; captions are keyed "
                             "by start frame")
        if t1 <= t0:
            raise ValueError(f"segment ({s1}, {t1}) lies within the previous segment, "
                             f"which ends at {t0}; it has no frames of its own")
    return spans


def caption_windows(video, output, segments,
                    n_sample=NUM_FRAMES, model_path=None, device="cuda"):
    """Caption every segment in `segments`, writing {start_frame: caption} to `output`.

    `segments` is `[(s1, t1), (s2, t2), ...]` of half-open output-frame ranges at
    FPS: segment k spans [s_k, t_k), and that is also the window it is generated
    in. Each segment's caption is sampled from the part of it that is actually
    generated: if segment k starts before segment k-1 ends, those overlapping
    head frames are pinned to the previous segment's output (retake), so they are
    skipped -- captioning them describes frames this segment does not draw, and
    since they are also the previous segment's tail they drag its state into this
    caption. The overlap is read off the segments themselves, so there is no
    separate window length or context to keep in sync with the schedule.

    The json is rewritten after every segment, so a crash keeps what is done.
    """
    spans = check_segments(segments)
    frames, src_fps = read_all(video)
    total = available_frames(frames, src_fps)
    print(f"{os.path.basename(video)}: {frames.shape[0]} src frames @ {src_fps:g}fps "
          f"-> {total} frames @ {FPS}fps; {len(spans)} segment(s) to caption", flush=True)

    model, processor = load_captioner(model_path, device)
    print("captioner loaded", flush=True)

    caps, failed, global_caption = {}, [], None

    def flush():
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w") as fh:
            json.dump({"segments": [list(p) for p in spans], "fps": FPS,
                       "global_caption": global_caption, "captions": caps},
                      fh, ensure_ascii=False, indent=1)

    gimgs, gspan = segment_frames(frames, src_fps, 0, total, n_sample, total)
    graw = caption_global(model, processor, gimgs, gspan / FPS)
    gcap = parse_output(graw)
    if gcap is None or "global_caption" not in gcap:
        raise RuntimeError(f"global caption parse failed -> {graw[:300]}")
    global_caption = str(gcap["global_caption"]).strip()
    print(f"[global] ok ({len(global_caption)} chars, {len(gimgs)} frames over "
          f"{gspan})\n{global_caption}\n", flush=True)

    # Always global then per-segment local (including single-segment short clips);
    # the generation prompt uses the local caption.
    prev_end = None
    for j, (s, t) in enumerate(spans):
        c0 = s if prev_end is None else max(s, prev_end)
        prev_end = t
        imgs, span = segment_frames(frames, src_fps, c0, t, n_sample, total)
        # Seconds are the real sampled interval, so `span` rather than the
        # nominal t - c0: the last segment gets clamped to what the file covers.
        raw = caption_frames(model, processor, imgs, span / FPS, global_caption=global_caption,
                             position={"total_dur": f"{total / FPS:.1f}", "start": f"{c0 / FPS:.1f}",
                                       "end": f"{(c0 + span) / FPS:.1f}", "index": j + 1, "count": len(spans)})
        cap = parse_output(raw)
        if cap is None or "local_caption" not in cap:
            failed.append(s)
            print(f"[{j+1}/{len(spans)}] segment=({s}, {t}) PARSE FAIL: {raw[:160]}", flush=True)
            continue
        caps[str(s)] = str(cap["local_caption"]).strip()
        print(f"[{j+1}/{len(spans)}] segment=({s}, {t}) ok ({len(caps[str(s)])} chars, "
              f"{len(imgs)} frames over [{c0}, {c0 + span}))", flush=True)
        flush()

    flush()
    print(f"wrote {len(caps)}/{len(spans)} captions -> {output}"
          + (f"  FAILED: {failed}" if failed else ""), flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} segment(s) failed to parse")
    return caps
