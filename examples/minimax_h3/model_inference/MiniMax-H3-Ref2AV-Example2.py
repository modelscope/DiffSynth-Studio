import os
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

import av
import torch
import torchaudio

from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

MODEL_ID = "MiniMaxAI/MiniMax-H3"
FPS = 24


def align_frame_count(frame_count):
    """Snap up to MiniMax-H3's 17n+5 frame grid.

    The pipeline applies this same alignment to `num_frames` internally, so the
    reference media has to be prepared at the aligned length to stay in sync
    with the generated clip.
    """
    current = max(int(frame_count), 1)
    while current % 17 != 5:
        current += 1
    return current


# 120 -> 124 frames (17*7+5 = 5.1667s).
NUM_FRAMES = 120
ALIGNED_FRAMES = align_frame_count(NUM_FRAMES)
AUDIO_SR = 44100

vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="Ref2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="Ref2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="Ref2VA/audio_vae/model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id=MODEL_ID, origin_file_pattern="Ref2VA/tokenizer/"),
    processor_config=ModelConfig(model_id=MODEL_ID, origin_file_pattern="Ref2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)


def read_video_24fps(path, num_out_frames):
    """Decode a video and resample it onto the 24fps time grid.

    The pipeline assumes reference frame lists are already 24fps CFR, so the fps
    conversion belongs here. Output frame k takes the source frame nearest to
    time k/24, matching what ffmpeg's `fps=24` filter picks.
    """
    container = av.open(path)
    src_fps = float(container.streams.video[0].average_rate)
    frames = [frame.to_image() for frame in container.decode(video=0)]
    container.close()
    return [
        frames[min(int(round(k * src_fps / FPS)), len(frames) - 1)]
        for k in range(num_out_frames)
    ]


def read_audio_stereo(path, num_out_frames):
    """Load a soundtrack, force stereo 44100Hz, and cut it to the video duration."""
    waveform, sr = torchaudio.load(path)
    if sr != AUDIO_SR:
        waveform = torchaudio.transforms.Resample(sr, AUDIO_SR)(waveform)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    waveform = waveform[:2]
    samples = int(round(num_out_frames / FPS * AUDIO_SR))
    if waveform.shape[-1] < samples:
        raise ValueError(f"{path} is shorter than {num_out_frames / FPS:.3f}s")
    return waveform[:, :samples]


# Example 2: silent motion/timing reference video + a separate voice reference.
# The video's own soundtrack is intentionally NOT used here, so it goes in as a
# `video` (silent) reference while voice.mp3 is a standalone `audio` reference.
ref_video = read_video_24fps("assets_minimax/ref2av/example2/video.mp4", ALIGNED_FRAMES)
ref_voice = read_audio_stereo("assets_minimax/ref2av/example2/voice.mp3", ALIGNED_FRAMES)
prompt = open("assets_minimax/ref2av/example2/prompt.txt").read().strip()

video, audio = pipe(
    prompt=prompt,
    height=768, width=1344, num_frames=NUM_FRAMES,
    num_inference_steps=50, seed=42,
    references=[
        {"type": "video", "video": ref_video},
        {"type": "audio", "audio": ref_voice, "sample_rate": AUDIO_SR},
    ],
)
write_video_audio(
    video=video,
    audio=audio,
    output_path="minimax_h3_ref2av_ex2.mp4",
    fps=FPS,
    audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_ref2av_ex2.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
