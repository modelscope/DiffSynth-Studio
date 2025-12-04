import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)


dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=f"data/examples/wan/input_image.jpg"
)
input_image = Image.open("data/examples/wan/input_image.jpg")

video = pipe(
    prompt="一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    input_image=input_image,
    camera_control_direction="Left", camera_control_speed=0.01,
)
save_video(video, "video_left_Wan2.1-Fun-V1.1-14B-Control-Camera.mp4", fps=15, quality=5)

video = pipe(
    prompt="一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    input_image=input_image,
    camera_control_direction="Up", camera_control_speed=0.01,
)
save_video(video, "video_up_Wan2.1-Fun-V1.1-14B-Control-Camera.mp4", fps=15, quality=5)
