import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
from diffsynth.utils.data.audio import read_audio
from diffsynth.utils.data import VideoData
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": torch.float8_e5m2,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e5m2,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    tokenizer_config=ModelConfig(path="./models/DiffSynth-Studio/LTX-2.5-Repackage/tokenizer"),
    model_configs=[
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2.5-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-video-vae-bf16.safetensors", **vram_config),
        # For lower VRAM and faster decoding, replace the line above with the conv vae decoder:
        # ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-video-vae-conv-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-audio-vae-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", **vram_config),
    ],
    stage2_lora_config=ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"),
    stage2_lora_strength=1.0,
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

dataset_snapshot_download("DiffSynth-Studio/example_video_dataset", allow_file_pattern="ltx2/*", local_dir="data/example_video_dataset")
prompt = "A beautiful woman with a flower crown is singing happily under a blooming cherry tree. She sings: 'Mummy don't know daddy's getting hot. At the body shop'"
negative_prompt = pipe.default_negative_prompt["LTX-2.5"]

height, width, num_frames, frame_rate = 512 * 2, 768 * 2, 121, 24
path = "data/example_video_dataset/ltx2/video2.mp4"
video = VideoData(path, height=height, width=width).raw_data()[:num_frames]
assert len(video) == num_frames, f"Input video has {len(video)} frames, but expected {num_frames} frames based on the specified num_frames argument."
audio, audio_sample_rate = read_audio(path)

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
    tile_size_in_frames=80,
    use_two_stage_pipeline=True,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path="ltx2.5_twostage_retake.mp4",
    fps=frame_rate,
    audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
)
