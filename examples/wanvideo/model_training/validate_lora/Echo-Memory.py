import torch
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.pipelines.wan_video_echo_memory import load_echo_memory_dit
from diffsynth.utils.data import save_video

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
load_echo_memory_dit(pipe)
pipe.load_lora(pipe.dit, "models/train/Echo-Memory_lora/epoch-4.safetensors", alpha=1)
pipe.enable_vram_management(num_persistent_param_in_dit=None)
video = pipe(
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔有几朵野花点缀，远处隐约可见蓝天和树木。小狗奔跑时四肢舒展，尾巴微微上扬，展现出轻快灵动的姿态。中景跟拍视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0,
    tiled=True,
)
save_video(video, "video_Echo-Memory_lora.mp4", fps=15, quality=5)
