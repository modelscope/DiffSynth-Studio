import torch, os
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/WanToDance-14B", origin_file_pattern="local_model.safetensors"),
        ModelConfig(model_id="Wan-AI/WanToDance-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/WanToDance-14B", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/WanToDance-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)
pipe.load_lora(pipe.dit, "models/train/WanToDance-14B-global_lora/epoch-4.safetensors", alpha=1)
dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="wanvideo/WanToDance-14B-local/*"
)
# This is a specialized model with the following constraints on its input parameters:
# *   The model renders and outputs video based on a sequence of keyframes; therefore, `wantodance_keyframes` must be provided correctly.
# *   If you need to generate a long video, please generate it in segments, and ensure that `wantodance_music_path`, `wantodance_keyframes`, and `wantodance_keyframes_mask` are properly split accordingly.
# *   The audio file specified by `wantodance_music_path` must match the video duration, calculated as (`num_frames` / 30) seconds.
# *   The width and height of `wantodance_reference_image` must be multiples of 16.
# *   `wantodance_fps` is configurable, but since the model appears to have been trained exclusively at 30 FPS, setting it to other values is not recommended.
# *   In `wantodance_keyframes`, frames that are not keyframes should be solid black.
# *   `wantodance_keyframes_mask` indicates the positions of valid frames within `wantodance_keyframes`.
wantodance_keyframes = VideoData("data/diffsynth_example_dataset/wanvideo/WanToDance-14B-local/keyframes.mp4")
wantodance_keyframes = [wantodance_keyframes[i] for i in range(149)]
video = pipe(
    prompt="一个人正在跳舞，舞蹈种类是古典舞,图像清晰程度高,人物动作平均幅度中等,人物动作最大幅度中等。, 帧率是30fps。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=1280, width=720, num_frames=149,
    num_inference_steps=24,
    wantodance_music_path="data/diffsynth_example_dataset/wanvideo/WanToDance-14B-local/music.wav",
    wantodance_reference_image=Image.open("data/diffsynth_example_dataset/wanvideo/WanToDance-14B-local/refimage.jpg"),
    wantodance_fps=30,
    wantodance_keyframes=wantodance_keyframes,
    wantodance_keyframes_mask=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1],
)
save_video(video, "video_WanToDance-14B-local.mp4", fps=30, quality=5)
