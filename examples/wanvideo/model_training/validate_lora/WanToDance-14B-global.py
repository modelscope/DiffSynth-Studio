import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/WanToDance-14B", origin_file_pattern="global_model.safetensors"),
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
    allow_file_pattern="wanvideo/WanToDance-14B-global/*"
)
# This is a specialized model with the following constraints on its input parameters:
# *   The model outputs a sequence of keyframes rather than a video; therefore, `framewise_decoding=True` must be set.
# *   When the number of keyframes is $n$, `num_frames` = 4 * (n - 1) + 1.
# *   Reducing `height`, `width`, `num_frames`, or `num_inference_steps` may lead to severe artifacts or generation failure.
# *   The audio file specified by `wantodance_music_path` must match the video duration, calculated as (`num_frames` / 7.5) seconds.
# *   The width and height of `wantodance_reference_image` must be multiples of 16.
# *   `wantodance_fps` is configurable, but since the model appears to have been trained exclusively at 7.5 FPS, setting it to other values is not recommended.
# *   The first frame of `wantodance_keyframes` is the `wantodance_reference_image`, while all subsequent frames are solid black.
# *   `wantodance_keyframes_mask` indicates the positions of valid frames within `wantodance_keyframes`.
wantodance_keyframes = VideoData("data/diffsynth_example_dataset/wanvideo/WanToDance-14B-global/keyframes.mp4")
wantodance_keyframes = [wantodance_keyframes[i] for i in range(149)]
video = pipe(
    prompt="一个人正在跳舞，舞蹈种类是韩舞。帧率是7.5000",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=False,
    height=1280, width=720, num_frames=149,
    num_inference_steps=48,
    wantodance_music_path="data/diffsynth_example_dataset/wanvideo/WanToDance-14B-global/music.WAV",
    wantodance_reference_image=Image.open("data/diffsynth_example_dataset/wanvideo/WanToDance-14B-global/refimage.jpg"),
    wantodance_fps=7.5,
    wantodance_keyframes=wantodance_keyframes,
    wantodance_keyframes_mask=[1] + [0] * 148,
    framewise_decoding=True,
)
save_video(video, "video_WanToDance-14B-global.mp4", fps=7.5, quality=5)
