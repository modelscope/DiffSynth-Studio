# This script can generate a single video clip.
# If you need generate long videos, please refer to `Wan2.2-S2V-14B_multi_clips.py`.
import torch
from PIL import Image
import librosa
from diffsynth.utils.data import VideoData, save_video_with_audio
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    audio_processor_config=ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/"),
)
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_video_dataset",
    local_dir="./data/example_video_dataset",
    allow_file_pattern=f"wans2v/*"
)

num_frames = 81 # 4n+1
height = 448
width = 832

prompt = "a person is singing"
negative_prompt = "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
input_image = Image.open("data/example_video_dataset/wans2v/pose.png").convert("RGB").resize((width, height))
# s2v audio input, recommend 16kHz sampling rate
audio_path = 'data/example_video_dataset/wans2v/sing.MP3'
input_audio, sample_rate = librosa.load(audio_path, sr=16000)

# Speech-to-video
video = pipe(
    prompt=prompt,
    input_image=input_image,
    negative_prompt=negative_prompt,
    seed=0,
    num_frames=num_frames,
    height=height,
    width=width,
    audio_sample_rate=sample_rate,
    input_audio=input_audio,
    num_inference_steps=40,
)
save_video_with_audio(video[1:], "video_1_Wan2.2-S2V-14B.mp4", audio_path, fps=16, quality=5)

# s2v will use the first (num_frames) frames as reference. height and width must be the same as input_image. And fps should be 16, the same as output video fps.
pose_video_path = 'data/example_video_dataset/wans2v/pose.mp4'
pose_video = VideoData(pose_video_path, height=height, width=width)

# Speech-to-video with pose
video = pipe(
    prompt=prompt,
    input_image=input_image,
    negative_prompt=negative_prompt,
    seed=0,
    num_frames=num_frames,
    height=height,
    width=width,
    audio_sample_rate=sample_rate,
    input_audio=input_audio,
    s2v_pose_video=pose_video,
    num_inference_steps=40,
)
save_video_with_audio(video[1:], "video_2_Wan2.2-S2V-14B.mp4", audio_path, fps=16, quality=5)
