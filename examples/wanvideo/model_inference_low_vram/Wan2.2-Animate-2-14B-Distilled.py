import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-2-14B", origin_file_pattern="wan_animate_2/wan_animate_2_bf16_distillation.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-2-14B", origin_file_pattern="videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-2-14B", origin_file_pattern="videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.2-Animate-2-14B", origin_file_pattern="videomodel/Wan-AI/umt5-xxl/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Character animation: reference image (identity) + reference video (motion) -> animated video.
dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="wanvideo/Wan2.2-Animate-2-14B-Distilled/*"
)
reference_image = Image.open("data/diffsynth_example_dataset/wanvideo/Wan2.2-Animate-2-14B-Distilled/refimage.jpg").convert("RGB")
reference_video = VideoData("data/diffsynth_example_dataset/wanvideo/Wan2.2-Animate-2-14B-Distilled/refvideo.mp4").raw_data()

# Example 1: single-clip generation
num_frames = 81
# For distilled model, set animate2_log_scale to -1.3, num_inference_steps to 10, and cfg_scale to 1.0.
video = pipe(
    prompt="人物外观描述：一名长黑发女性，穿着白色半透明蕾丝长袖上衣，衣身带有花卉刺绣，下身搭配白色百褶短裙和黑色腰带，脚穿米白色厚底运动鞋。 背景描述：背景为现代室内空间，墙面和柜体以浅灰色为主，后方设有两扇深色落地窗或玻璃门，顶部安装长条形灯具，中央有一块浅色长方形台面。",
    animate2_prompt_ref="视频中的人在做动作，背景静止",
    animate2_reference_image=reference_image,
    animate2_reference_video=reference_video[:num_frames],
    animate2_offload_kv=True,
    animate2_log_scale=-1.3,
    num_frames=num_frames, height=1280, width=720,
    num_inference_steps=10, cfg_scale=1.0,
    seed=0, tiled=True,
)
save_video(video, "video_Wan2.2-Animate-2-14B-Distilled.mp4", fps=24, quality=5)


# Example 2: multi-clip long-video generation
def generate_long_video(pipe, reference_image, cond_images, clip_len, first_num=1, **kwargs):
    assert clip_len > first_num, "clip_len must be greater than first_num"

    def zigzag_padding(array, target_len):
        if len(array) == 1:
            return [array[0]] * target_len
        idx, flip, out = 0, False, []
        while len(out) < target_len:
            out.append(array[idx])
            idx += -1 if flip else 1
            if idx == 0 or idx == len(array) - 1:
                flip = not flip
        return out[:target_len]

    real_len = len(cond_images)
    if real_len == 0:
        return []
    step = clip_len - first_num
    # Precompute clip count so clips of `clip_len` stepping by `step` tile the (padded) driving video.
    num_clips = 1 if real_len <= clip_len else (real_len - clip_len + step - 1) // step + 1
    target_len = clip_len + (num_clips - 1) * step
    if real_len < target_len:
        cond_images = zigzag_padding(cond_images, target_len)

    all_frames = []
    prev_tail = None
    for i in range(num_clips):
        start = i * step
        seg_driving = cond_images[start:start + clip_len]
        seg_out = pipe(
            animate2_reference_image=reference_image,
            animate2_reference_video=seg_driving,
            animate2_refert_images=None if i == 0 else prev_tail,
            num_frames=clip_len,
            **kwargs,
        )
        prev_tail = seg_out[-first_num:]
        if i != 0:
            seg_out = seg_out[first_num:]
        all_frames.extend(seg_out)
    return all_frames[:real_len]


clip_len = 81
# For distilled model, set animate2_log_scale to -1.3, num_inference_steps to 10, and cfg_scale to 1.0.
long_video = generate_long_video(
    pipe,
    reference_image=reference_image,
    cond_images=reference_video,
    clip_len=clip_len,
    first_num=1,
    prompt="人物外观描述：一名长黑发女性，穿着白色半透明蕾丝长袖上衣，衣身带有花卉刺绣，下身搭配白色百褶短裙和黑色腰带，脚穿米白色厚底运动鞋。 背景描述：背景为现代室内空间，墙面和柜体以浅灰色为主，后方设有两扇深色落地窗或玻璃门，顶部安装长条形灯具，中央有一块浅色长方形台面。",
    animate2_prompt_ref="视频中的人在做动作，背景静止",
    animate2_offload_kv=True,
    animate2_log_scale=-1.3,
    height=1280, width=720,
    num_inference_steps=10, cfg_scale=1.0,
    seed=0, tiled=True,
)
save_video(long_video, "video_Wan2.2-Animate-2-14B-Distilled-long.mp4", fps=24, quality=5)
