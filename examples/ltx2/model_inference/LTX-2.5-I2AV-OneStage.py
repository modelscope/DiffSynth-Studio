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
    tokenizer_config=ModelConfig(path="./models/DiffSynth-Studio/LTX-2.5-Repackage/tokenizer"),
    model_configs=[
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2.5-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-video-vae-bf16.safetensors", **vram_config),
        # For lower VRAM and faster decoding, replace the line above with the conv vae decoder:
        # ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-video-vae-conv-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-audio-vae-bf16.safetensors", **vram_config),
    ],
)
dataset_snapshot_download("DiffSynth-Studio/example_video_dataset", allow_file_pattern="ltx2/*", local_dir="data/example_video_dataset")
# The example image comes from the shared sample dataset, so reuse its paired prompt.
prompt = "A beautiful woman with a flower crown is singing happily under a blooming cherry tree. She sings: 'Mummy don't know daddy's getting hot. At the body shop'"
negative_prompt = pipe.default_negative_prompt["LTX-2.5"]
height, width, num_frames = 512 * 2, 768 * 2, 121
first_frame = Image.open("data/example_video_dataset/ltx2/first_frame.png").convert("RGB").resize((width, height))
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=42,
    height=height,
    width=width,
    num_frames=num_frames,
    tiled=True,
    tile_size_in_frames=80,
    cfg_scale=3.0,
    input_images=[first_frame],
    input_images_indexes=[0],
    input_images_strength=1.0,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path="ltx2.5_onestage_i2av_first.mp4",
    fps=24,
    audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
)
