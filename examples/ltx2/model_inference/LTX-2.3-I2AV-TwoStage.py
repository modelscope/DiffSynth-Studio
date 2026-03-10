import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
from PIL import Image
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
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
)

prompt = "Two cute orange cats, wearing boxing gloves, stand in a boxing ring and fight each other. They are punching each other fast and yelling: 'I will win!'"
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
height, width, num_frames = 512 * 2, 768 * 2, 121
dataset_snapshot_download("DiffSynth-Studio/example_video_dataset", allow_file_pattern="ltx2/*", local_dir="data/example_video_dataset")
first_frame = Image.open("data/example_video_dataset/ltx2/first_frame.png").convert("RGB").resize((width, height))
last_frame = Image.open("data/example_video_dataset/ltx2/last_frame.png").convert("RGB").resize((width, height))
# first frame
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=42,
    height=height,
    width=width,
    num_frames=num_frames,
    tiled=True,
    use_two_stage_pipeline=True,
    input_images=[first_frame],
    input_images_indexes=[0],
    input_images_strength=1.0,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path='ltx2.3_twostage_i2av_first.mp4',
    fps=24,
)
pipe.clear_lora()

# This example uses the first and last frames for demonstration. However, you can use any frames by setting input_images and input_indexes. Note that input_indexes must be within the range of num_frames.
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=42,
    height=height,
    width=width,
    num_frames=num_frames,
    tiled=True,
    use_two_stage_pipeline=True,
    input_images=[first_frame, last_frame],
    input_images_indexes=[0, num_frames-1],
    input_images_strength=1.0,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path='ltx2.3_twostage_i2av_first_last.mp4',
    fps=24,
)
