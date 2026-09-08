import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data import VideoData
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
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
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-video-vae-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-audio-vae-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
pipe.load_lora(
    pipe.dit,
    ModelConfig(
        model_id="Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler",
        origin_file_pattern="ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
    ),
)

dataset_snapshot_download("DiffSynth-Studio/example_video_dataset", allow_file_pattern="ltx2/*", local_dir="data/example_video_dataset")
# The reference video comes from the shared sample dataset, so reuse its paired prompt.
prompt = "A beautiful woman with a flower crown is singing happily under a blooming cherry tree. She sings: 'Mummy don't know daddy's getting hot. At the body shop'"
negative_prompt = pipe.default_negative_prompt["LTX-2.5"]
height, width, num_frames = 512 * 2, 768 * 2, 121
reference_video = VideoData("data/example_video_dataset/ltx2/video2.mp4", height=height // 4, width=width // 4).raw_data()
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=43,
    height=height,
    width=width,
    num_frames=num_frames,
    frame_rate=24,
    in_context_videos=[reference_video],
    in_context_downsample_factor=2,
    tiled=True,
    cfg_scale=1.0,
    num_inference_steps=8,
    use_distilled_pipeline=True,
    use_two_stage_pipeline=True,
    clear_lora_before_state_two=True,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path="ltx2.5_pixel_spatial_upscale.mp4",
    fps=24,
    audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
)
