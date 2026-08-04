import torch
from PIL import Image
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
pipe.load_lora(pipe.dit, "models/train/MiniMax-H3-Ref2VA-split/epoch-4.safetensors")

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="minimax_h3/MiniMax-H3-Ref2VA/*",
)
dataset_base_path = "data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA"
height, width, num_frames = 480, 832, 124
ref_image = Image.open(f"{dataset_base_path}/0.png").convert("RGB")
prompt = "一个网站页面，网站页面UI设计，网站动效，视频展示了流畅的网页向下滚动效果。一个极具爆发力与动感的产品官网风格产品落地页 UI/UX 演示视频，核心展示主体是该产品图片1。页面采用粗犷有力、倾斜的超大号无衬线字体进行张扬的排版。背景有极具速度感的动态光影、暗色碳纤维或运动透气网眼纹理在交织变换。视频展示了节奏紧凑、充满力量感的网页向下滚动效果，以及鼠标悬停时强烈的视觉放大与颜色反转等 UI 交互动作。"

video, audio = pipe(
    prompt=prompt,
    height=height, width=width, num_frames=num_frames,
    num_inference_steps=50, seed=42,
    references=[{"type": "image", "image": ref_image}],
)
write_video_audio(
    video=video, audio=audio, output_path="minimax_h3_ref2va_lora.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_ref2va_lora.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
