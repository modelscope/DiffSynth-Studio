import os
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")  # env flash_attn is a stub
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")  # load pre-downloaded local files

import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

MODEL_ID = "MiniMaxAI/MiniMax-H3"

# Low-VRAM inference: disk offload + bf16. Params rest on disk, stream through
# CPU (bf16) to CUDA. Only components with a registered module_map (DiT + text
# encoder) take the fine-grained vram_config. bf16 onload keeps weight dtype ==
# compute dtype, so the fp32-whitelist projections stay consistent.
vram_config = {
    "offload_dtype": "disk", "offload_device": "disk",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/video_vae/source/model.safetensors"),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern="FL2VA/audio_vae/model.safetensors"),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

video, audio = pipe(
    prompt="实拍深夜自助洗衣店场景，融合手绘发光动画的混合影像。一间小型自助洗衣店，荧光灯微微频闪，店内摆放着运转中的洗衣机、塑料洗衣篮、老旧长椅，地面散落一只袜子，整体空间安静，带着淡淡的怀旧氛围感。手持手机单手拍摄质感，画面抖动明显；白色荧光灯造成曝光忽明忽暗；玻璃表面带有环境反光；镜头靠近物体时对焦存在延迟。画面不要像商业广告一样精致规整，整体质感如同深夜偶然闯入、追着奇异幻象随手拍下的真实纪实感。",
    height=768, width=1344, num_frames=24,
    num_inference_steps=50, seed=42,
)
write_video_audio(
    video=video,
    audio=audio,
    output_path="minimax_h3_t2va_low_vram.mp4",
    fps=24,
    audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_t2va_low_vram.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
