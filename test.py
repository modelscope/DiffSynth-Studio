import torch
torch.cuda.set_per_process_memory_fraction(0.999, 0)
from diffsynth import ModelManager, save_video, VideoData, save_frames, save_video, download_models
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, model_fn_wan_video
from diffsynth.controlnets.processors import Annotator
from diffsynth.data.video import crop_and_resize
from modelscope import snapshot_download
from tqdm import tqdm
from PIL import Image


# Load models
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        # ModelConfig("D:\projects\VideoX-Fun\models\Wan2.1-Fun-V1.1-1.3B-Control\diffusion_pytorch_model.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)

video = VideoData(rf"D:\pr_projects\20250503_dance\data\双马尾竖屏暴击！你的微笑就是彩虹的微笑♥ - 1.双马尾竖屏暴击！你的微笑就是彩虹的微笑♥(Av114086629088385,P1).mp4", height=832, width=480)
annotator = Annotator("openpose")
video = [video[i] for i in tqdm(range(450, 450+1*81, 1))]
save_video(video, "video_input.mp4", fps=60, quality=5)
control_video = [annotator(f)  for f in tqdm(video)]
save_video(control_video, "video_control.mp4", fps=60, quality=5)
reference_image = crop_and_resize(Image.open(rf"D:\pr_projects\20250503_dance\data\marmot4.png"), 832, 480)

with torch.amp.autocast("cuda", torch.bfloat16):
    video = pipe(
        prompt="微距摄影风格特写画面，一只憨态可掬的土拨鼠正用后腿站立在碎石堆上，它在挥舞着双臂。金棕色的绒毛在阳光下泛着丝绸般的光泽，腹部毛发呈现浅杏色渐变，每根毛尖都闪烁着细密的光晕。两只黑曜石般的眼睛透出机警而温顺的光芒，鼻梁两侧的白色触须微微颤动，捕捉着空气中的气息。背景是虚化的灰绿色渐变，几簇嫩绿苔藓从画面右下角探出头来，与前景散落的鹅卵石形成微妙的景深对比。土拨鼠圆润的身形在逆光中勾勒出柔和的轮廓，耳朵紧贴头部的姿态流露出戒备中的天真，整个画面洋溢着自然界生灵特有的灵动与纯真。",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        seed=43, tiled=True,
        height=832, width=480, num_frames=len(control_video),
        control_video=control_video, reference_image=reference_image,
        # sliding_window_size=5, sliding_window_stride=2,
        # num_inference_steps=100,
        # cfg_merge=True,
        sigma_shift=16,
    )
    save_video(video, "video1.mp4", fps=60, quality=5)
