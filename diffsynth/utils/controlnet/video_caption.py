"""Per-window captions for one long video, for window-by-window generation.

Handing every window the same whole-clip caption tells each one to perform the
entire arc, so a clip with a one-way trajectory (a flower opening) snaps back to
the start of that arc at every seam. One caption per window instead: segment k is
told what happens in segment k's own span. Measured on a 40s bloom with 56-frame
windows, this roughly halves the mean seam discontinuity at every context length.

The whole clip is captioned first; each window then gets its own local caption
with that global text as a naming reference only (including single-window short
clips). Generation consumes the local caption. Handing a window its
*predecessor's* caption instead collapses the output -- greedy decoding copies
a caption already nearly right for the shared frames.

Windows are identified by their start frame g0 in output-FPS index space, so the
caller owns the schedule and there is exactly one definition of it.

Run as a subprocess (`python -m diffsynth.utils.controlnet.video_caption in.mp4
out.json --g0 0,344 --window 345 --context 1`) when a generation pipeline
follows: the captioner is a 27B model and only process exit releases the GPU.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

MODEL_PATH = "/root/models/Qwen3.8-27B"
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


def window_frames(frames, src_fps, g0, window, n_sample, total):
    """Sample up to n_sample frames out of output-index window [g0, g0+window).

    Output frame k maps to source frame round(k*src_fps/FPS), the same mapping
    the generation loader uses; sampling the source uniformly instead would
    drift against the frames the model actually renders for this window.

    The span is clamped to what the file covers, and if that leaves fewer than
    n_sample frames, fewer are sampled rather than duplicated. A schedule whose
    last window overruns the end would otherwise clamp most positions onto the
    final frame and get described as static.

    Returns (images, span) where span is how many output frames they really
    cover -- what the prompt should be told the duration is.
    """
    T = frames.shape[0]
    end = min(g0 + window - 1, total - 1)
    n = min(n_sample, end - g0 + 1)
    imgs = [_scale(Image.fromarray(frames[min(int(round(k * src_fps / FPS)), T - 1)]).convert("RGB"))
            for k in np.linspace(g0, end, n, dtype=int)]
    return imgs, end - g0 + 1


# ------------------------------------------------------------------- inference
def load_captioner(model_path=MODEL_PATH, device="cuda"):
    from transformers import AutoModelForImageTextToText, AutoProcessor
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
    """`global_caption` makes this window use the whole clip's naming;
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
def caption_windows(video, output, g0_list, window, context=0,
                    n_sample=NUM_FRAMES, model_path=MODEL_PATH, device="cuda"):
    """Caption every window in `g0_list`, writing {g0: caption} to `output`.

    Each window's caption is sampled from the part of it that is actually
    generated: with retake, the first `context` frames of every window after the
    first are pinned to the previous segment's output, so they are skipped --
    captioning them describes frames this segment does not draw, and since they
    are also the previous window's tail they drag its state into this caption.

    The json is rewritten after every window, so a crash keeps what is done.
    """
    want = sorted({int(x) for x in g0_list})
    frames, src_fps = read_all(video)
    total = available_frames(frames, src_fps)
    print(f"{os.path.basename(video)}: {frames.shape[0]} src frames @ {src_fps:g}fps "
          f"-> {total} frames @ {FPS}fps; {len(want)} windows of {window} to caption", flush=True)

    model, processor = load_captioner(model_path, device)
    print("captioner loaded", flush=True)

    caps, failed, global_caption = {}, [], None

    def flush():
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w") as fh:
            json.dump({"window": window, "fps": FPS, "global_caption": global_caption,
                       "captions": caps}, fh, ensure_ascii=False, indent=1)

    gimgs, gspan = window_frames(frames, src_fps, 0, total, n_sample, total)
    graw = caption_global(model, processor, gimgs, gspan / FPS)
    gcap = parse_output(graw)
    if gcap is None or "global_caption" not in gcap:
        raise RuntimeError(f"global caption parse failed -> {graw[:300]}")
    global_caption = str(gcap["global_caption"]).strip()
    print(f"[global] ok ({len(global_caption)} chars, {len(gimgs)} frames over "
          f"{gspan})\n{global_caption}\n", flush=True)

    # Always global then per-window local (including single-window short clips);
    # the generation prompt uses the local caption.
    for j, g0 in enumerate(want):
        c0 = g0 if j == 0 else g0 + context
        imgs, span = window_frames(frames, src_fps, c0, window - (c0 - g0), n_sample, total)
        # Seconds are the real sampled interval, so `span` rather than `window`:
        # the last window gets clamped to what the file covers.
        raw = caption_frames(model, processor, imgs, span / FPS, global_caption=global_caption,
                             position={"total_dur": f"{total / FPS:.1f}", "start": f"{c0 / FPS:.1f}",
                                       "end": f"{(c0 + span) / FPS:.1f}", "index": j + 1, "count": len(want)})
        cap = parse_output(raw)
        if cap is None or "local_caption" not in cap:
            failed.append(g0)
            print(f"[{j+1}/{len(want)}] g0={g0} PARSE FAIL: {raw[:160]}", flush=True)
            continue
        caps[str(g0)] = str(cap["local_caption"]).strip()
        print(f"[{j+1}/{len(want)}] g0={g0} ok ({len(caps[str(g0)])} chars, "
              f"{len(imgs)} frames over {span})", flush=True)
        flush()

    flush()
    print(f"wrote {len(caps)}/{len(want)} captions -> {output}"
          + (f"  FAILED: {failed}" if failed else ""), flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} window(s) failed to parse")
    return caps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("output")
    p.add_argument("--g0", required=True, help="comma-separated window start frames")
    p.add_argument("--window", type=int, required=True, help="frames per window")
    p.add_argument("--context", type=int, default=0, help="pinned tail from the previous segment, skipped when sampling")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES, help=f"frames sampled per window (default {NUM_FRAMES})")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    caption_windows(args.video, args.output, [x for x in args.g0.split(",") if x.strip()],
                    args.window, context=args.context, n_sample=args.num_frames,
                    model_path=args.model_path, device=args.device)


if __name__ == "__main__":
    main()
