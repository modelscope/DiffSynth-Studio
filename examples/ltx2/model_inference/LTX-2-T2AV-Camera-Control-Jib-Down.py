import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

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
        ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-spatial-upscaler-x2-1.0.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
    stage2_lora_config=ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-distilled-lora-384.safetensors"),
)
pipe.load_lora(
    pipe.dit,
    ModelConfig(model_id="Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down", origin_file_pattern="ltx-2-19b-lora-camera-control-jib-down.safetensors"),
)
prompt = (
    "A girl is very happy, standing on a clean studio floor with soft ambient lighting. "
    "She is speaking directly to the camera: “I enjoy working with Diffsynth-Studio, it's a perfect framework.” "
    "The shot begins with a medium close-up framing her from the waist up. As the camera performs a smooth jib-down movement—"
    "descending vertically downward—it gradually reveals more of the lower portion of the scene. "
    "During the descent, the following elements become visible near the bottom of the frame: "
    "- The polished concrete floor with subtle reflections of the girl’s shoes, "
    "- A small branded mat labeled “Diffsynth-Studio” placed just beneath her feet, "
    "- The lower part of a sleek workstation desk with a glowing logo on its front panel, partially hidden at the start but fully revealed as the camera lowers. "
    "This downward motion provides a dynamic reveal of contextual details that reinforce the professional and creative environment, "
    "while maintaining focus on the girl’s enthusiastic expression throughout."
)
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
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=43,
    height=height,
    width=width,
    num_frames=num_frames,
    tiled=True,
    use_two_stage_pipeline=True,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path='ltx2_camera_jib_down.mp4',
    fps=24,
    audio_sample_rate=24000,
)
