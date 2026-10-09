import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from modelscope import dataset_snapshot_download
from PIL import Image

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
# The linear branch is copied into the attention wrappers at load time, so its
# parameters must be materialised (cpu), not left on the meta device by disk offload.
branch_vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cpu",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="OpenVDN/vdn-minimax-h3", origin_file_pattern="stage-b-step-2000/linear_branch/model.safetensors", **branch_vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

# Stage B trained this LoRA jointly with the linear branch; base + branch alone is not stage-b.
pipe.load_lora(pipe.dit, ModelConfig(model_id="OpenVDN/vdn-minimax-h3", origin_file_pattern="stage-b-step-2000/adapters/default/adapter_model.safetensors"))

# Text -> Video + Audio
prompt = "A girl is very happy, she is speaking in english: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
video, audio = pipe(
    prompt=prompt,
    height=768, width=1344, num_frames=345, num_inference_steps=50, seed=0,
    # upstream 50nfe.yaml: 50 model evaluations at 345 frames (14.4 s). The window is 15
    # latent frames wide, so the hybrid only pays off on long clips -- at 124 frames it is
    # ~1.2x over dense, at 345 it is ~2x, and at 480x832 it is slower than dense.
)
write_video_audio(
    video=video, audio=audio,
    output_path="vdn-t2va.mp4", fps=24, audio_sample_rate=32000,
)

# Text + First Frame + Last Frame -> Video + Audio
dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-FL2VA/*")
first_frame = Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/first.png")
last_frame = Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/last.png")
prompt = "室内家庭争吵短剧场景，竖屏短剧质感，真实真人表演，中式家庭/小饭馆室内环境，暖色灯光，背景有红色装饰和书法字幅，浅景深，情绪强烈，剪辑节奏紧凑。表演要求：真实短剧表演风格，不要夸张舞台腔。男人的语气是愤怒、委屈、急切的反驳，他说“你到底想干什么？”；中老年女性的语气是尖锐、强势、咄咄逼人的质问，她说“你必须赔钱！”。两人之间有强烈对峙感，节奏逐步升级。画面风格：竖屏9:16，手机短剧质感，真人实拍感，浅景深，室内暖光，中近景为主，频繁正反打剪辑，背景保持生活化，不要科幻、不要古装、不要动画感。画面中不要出现任何字幕、文字、平台水印或贴片。 "
video, audio = pipe(
    prompt=prompt,
    height=832, width=480, num_frames=345, num_inference_steps=50, seed=0,
    # upstream 50nfe.yaml: 50 model evaluations at 345 frames (14.4 s). The window is 15
    # latent frames wide, so the hybrid only pays off on long clips -- at 124 frames it is
    # ~1.2x over dense, at 345 it is ~2x, and at 480x832 it is slower than dense.
    keyframes=[first_frame, last_frame], keyframe_indices=[0, -1],
)
write_video_audio(
    video=video, audio=audio,
    output_path="vdn-fl2va.mp4", fps=24, audio_sample_rate=32000,
)
