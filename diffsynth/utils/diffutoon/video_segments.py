from PIL import Image
from tqdm import tqdm


def generate_segments(video_length, clip_length=328, clip_overlap=5):
    assert clip_length % 17 == 5
    segments = []
    for start in range(0, video_length, clip_length - clip_overlap):
        end = min(start + clip_length, video_length)
        if end - start < 17 + 5:
            break
        if (end - start) % 17 != 5:
            end = start + (end - start - 5) // 17 * 17 + 5
        segments.append((start, end))
    return segments


def process_long_video(pipe, outline_video, color_video, prompt, segments, seed=43, num_inference_steps=4, **pipe_kwargs):
    height, width = outline_video[0].height, outline_video[0].width
    frames = []
    pin_latents = None
    for clip_id, (start, end) in enumerate(tqdm(segments)):
        overlap = 0 if clip_id == 0 else segments[clip_id - 1][1] - start
        clip, audio, pin_latents = pipe(
            prompt=prompt if isinstance(prompt, str) else prompt[clip_id],
            height=height, width=width, num_frames=end-start,
            num_inference_steps=num_inference_steps, seed=seed + clip_id,
            control_video=[outline_video[start:end], color_video[start:end]], control_scale=[0.4, 0.4],
            pin_latents=pin_latents, return_latents=True,
            **pipe_kwargs,
        )
        pin_latents = pin_latents[:, :, -2:]
        overlap = 0 if clip_id == 0 else segments[clip_id - 1][1] - start
        if overlap:
            for i in range(overlap):
                frames[-overlap + i] = Image.blend(frames[-overlap + i], clip[i], i / max(overlap - 1, 1))
        frames.extend(clip[overlap:])
    return frames
