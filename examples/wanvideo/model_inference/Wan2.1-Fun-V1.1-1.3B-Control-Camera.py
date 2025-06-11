import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
from dchen.camera_compute import process_pose_file

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/control_video.mp4", "data/examples/wan/reference_image_girl.png"]
)

# Control video
control_video = None
reference_image = None
control_camera_text = "/mnt/nas2/dchen/Work/add_0609/DiffSynth-Studio/dchen/camera_information.txt"
input_image = Image.open("/mnt/nas2/dchen/Work/add_0609/DiffSynth-Studio/dchen/7.png")
sigma_shift = 3
height = 480
width = 832

control_camera_video = process_pose_file(control_camera_text, width, height)

video = pipe(
    prompt="一个小女孩正在户外玩耍。她穿着一件蓝色的短袖上衣和粉色的短裤，头发扎成一个可爱的辫子。她的脚上没有穿鞋，显得非常自然和随意。她正用一把红色的小铲子在泥土里挖土，似乎在进行某种有趣的活动，可能是种花或是挖掘宝藏。地上有一根长长的水管，可能是用来浇水的。背景是一片草地和一些绿色植物，阳光明媚，整个场景充满了童趣和生机。小女孩专注的表情和认真的动作让人感受到她的快乐和好奇心。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    control_video=control_video, reference_image=reference_image,
    height=height, width=width, num_frames=81,
    seed=1, tiled=True,

    control_camera_video = control_camera_video,
    input_image = input_image,
    sigma_shift = sigma_shift,
)
save_video(video, "video.mp4", fps=15, quality=5)
