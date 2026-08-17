import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.utils.data.audio_video import write_video_audio
from diffsynth.core.data.operators import ImageCropAndResize
from modelscope import snapshot_download
from PIL import Image

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cuda",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-fl2va-pruned-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=None,
)
pipe.load_lora(
    pipe.dit,
    ModelConfig(
        model_id="lightx2v/Minimax-h3-Turbo",
        origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
    ),
)
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(
        model_id="DiffSynth-Studio/MiniMax-H3-Text-Embeddings", origin_file_pattern="models/art_is_explosion/",
    )],
)
snapshot_download("DiffSynth-Studio/MiniMax-H3-Text-Embeddings", allow_file_pattern="assets/image_1.jpg", local_dir="data")
first_frame = ImageCropAndResize(height=1344, width=768)(Image.open("data/assets/image_1.jpg"))
video, audio = template(
    pipe,
    height=1344, width=768, num_frames=56,
    num_inference_steps=4, seed=0, flow_shift=6,
    keyframes=[first_frame], keyframe_indices=[0],
    template_inputs=[{}],
)
write_video_audio(
    video=video, audio=audio,
    output_path="output.mp4", fps=24, audio_sample_rate=32000,
)