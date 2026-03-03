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
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="transformer.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="video_vae_decoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="audio_vae_decoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="audio_vocoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="video_vae_encoder.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
)
prompt = "A girl is very happy, she is speaking: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
negative_prompt = "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
height, width, num_frames = 512 * 2, 768 * 2, 121
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/ltx-2/first_frame.jpg"]
)
image = Image.open("data/examples/ltx-2/first_frame.jpg").convert("RGB").resize((width, height))
# first frame
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=43,
    height=height,
    width=width,
    num_frames=num_frames,
    tiled=False,
    input_images=[image],
    input_images_indexes=[0],
    input_images_strength=1.0,
    num_inference_steps=40,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path='ltx2_onestage_i2av_first.mp4',
    fps=24,
    audio_sample_rate=24000,
)
