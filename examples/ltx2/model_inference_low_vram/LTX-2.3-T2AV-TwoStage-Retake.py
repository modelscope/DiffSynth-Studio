import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
from diffsynth.utils.data.audio import read_audio
from modelscope import dataset_snapshot_download
from diffsynth.utils.data import VideoData

vram_config = {
    "offload_dtype": torch.float8_e5m2,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e5m2,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e5m2,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-dev.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-spatial-upscaler-x2-1.0.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
    stage2_lora_config=ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-distilled-lora-384.safetensors"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

dataset_snapshot_download("DiffSynth-Studio/example_video_dataset", allow_file_pattern="ltx2/*", local_dir="data/example_video_dataset")
prompt = "A beautiful woman with a flower crown is singing happily under a blooming cherry tree. She sings: 'Mummy don't know daddy's getting hot. At the body shop'"
negative_prompt = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

height, width, num_frames, frame_rate = 512 * 2, 768 * 2, 121, 24
path = "data/example_video_dataset/ltx2/video2.mp4"
video = VideoData(path, height=height, width=width).raw_data()[:num_frames]
assert len(video) == num_frames, f"Input video has {len(video)} frames, but expected {num_frames} frames based on the specified num_frames argument."
duration = num_frames / frame_rate
audio, audio_sample_rate = read_audio(path)

# Regenerate the video within time regions. You can specify different time regions for video frames and audio retake.
# retake regions are in seconds, and the example below retakes video frames in the time regions of [1s, 2s] and [3s, 4s], and retakes audio in the time regions of [0s, 1s] and [4s, 5s].
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    retake_video=video,
    retake_video_regions=[(1, 2), (3, 4)],
    retake_audio=audio,
    audio_sample_rate=audio_sample_rate,
    retake_audio_regions=[(0, 1), (4, 5)],
    seed=43,
    height=height,
    width=width,
    num_frames=num_frames,
    frame_rate=frame_rate,
    tiled=True,
    use_two_stage_pipeline=True,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path='ltx2.3_twostage_retake.mp4',
    fps=frame_rate,
    audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
)
