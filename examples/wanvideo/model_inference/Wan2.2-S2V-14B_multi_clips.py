import torch
from PIL import Image
import librosa
from diffsynth import VideoData, save_video_with_audio
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, WanVideoUnit_S2V
from modelscope import dataset_snapshot_download


def speech_to_video(
    prompt,
    input_image,
    audio_path,
    negative_prompt="",
    num_clip=None,
    audio_sample_rate=16000,
    pose_video_path=None,
    infer_frames=80,
    height=448,
    width=832,
    num_inference_steps=40,
    fps=16, # recommend fixing fps as 16 for s2v
    motion_frames=73, # hyperparameter of wan2.2-s2v
    save_path=None,
):
    # s2v audio input, recommend 16kHz sampling rate
    input_audio, sample_rate = librosa.load(audio_path, sr=audio_sample_rate)
    # s2v will use the first (num_frames) frames as reference. height and width must be the same as input_image. And fps should be 16, the same as output video fps.
    pose_video = VideoData(pose_video_path, height=height, width=width) if pose_video_path is not None else None

    audio_embeds, pose_latents, num_repeat = WanVideoUnit_S2V.pre_calculate_audio_pose(
        pipe=pipe,
        input_audio=input_audio,
        audio_sample_rate=sample_rate,
        s2v_pose_video=pose_video,
        num_frames=infer_frames + 1,
        height=height,
        width=width,
        fps=fps,
    )
    num_repeat = min(num_repeat, num_clip) if num_clip is not None else num_repeat
    print(f"Generating {num_repeat} video clips...")
    motion_videos = []
    video = []
    for r in range(num_repeat):
        s2v_pose_latents = pose_latents[r] if pose_latents is not None else None
        current_clip = pipe(
            prompt=prompt,
            input_image=input_image,
            negative_prompt=negative_prompt,
            seed=0,
            num_frames=infer_frames + 1,
            height=height,
            width=width,
            audio_embeds=audio_embeds[r],
            s2v_pose_latents=s2v_pose_latents,
            motion_video=motion_videos,
            num_inference_steps=num_inference_steps,
        )
        current_clip = current_clip[-infer_frames:]
        if r == 0:
            current_clip = current_clip[3:]
        overlap_frames_num = min(motion_frames, len(current_clip))
        motion_videos = motion_videos[overlap_frames_num:] + current_clip[-overlap_frames_num:]
        video.extend(current_clip)
        save_video_with_audio(video, save_path, audio_path, fps=16, quality=5)
        print(f"processed the {r+1}th clip of total {num_repeat} clips.")
    return video


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    audio_processor_config=ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/"),
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_video_dataset",
    local_dir="./data/example_video_dataset",
    allow_file_pattern=f"wans2v/*",
)

infer_frames = 80  # 4n
height = 448
width = 832

prompt = "a person is singing"
negative_prompt = "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
input_image = Image.open("data/example_video_dataset/wans2v/pose.png").convert("RGB").resize((width, height))

video_with_audio = speech_to_video(
    prompt=prompt,
    input_image=input_image,
    audio_path='data/example_video_dataset/wans2v/sing.MP3',
    negative_prompt=negative_prompt,
    pose_video_path='data/example_video_dataset/wans2v/pose.mp4',
    save_path="video_with_audio_full.mp4",
    infer_frames=infer_frames,
    height=height,
    width=width,
)
# num_clip means generating only the first n clips with n * infer_frames frames.
video_with_audio_pose = speech_to_video(
    prompt=prompt,
    input_image=input_image,
    audio_path='data/example_video_dataset/wans2v/sing.MP3',
    negative_prompt=negative_prompt,
    pose_video_path='data/example_video_dataset/wans2v/pose.mp4',
    save_path="video_with_audio_pose_clip_2.mp4",
    num_clip=2
)
