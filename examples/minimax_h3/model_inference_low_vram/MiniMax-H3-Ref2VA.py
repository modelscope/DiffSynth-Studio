import torch
from PIL import Image
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio
from diffsynth.utils.data.audio import read_audio
from modelscope import dataset_snapshot_download

def align_frame_count(frame_count):
    current = max(int(frame_count), 1)
    while current % 17 != 5:
        current += 1
    return current

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
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)

# Text + Reference Image -> Video + Audio
dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Ref2VA/*")
ref_image = Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/0.png").convert("RGB")
prompt = "一个网站页面，网站页面UI设计，网站动效，视频展示了流畅的网页向下滚动效果。一个极具爆发力与动感的产品官网风格产品落地页 UI/UX 演示视频，核心展示主体是该产品图片1。页面采用粗犷有力、倾斜的超大号无衬线字体进行张扬的排版。背景有极具速度感的动态光影、暗色碳纤维或运动透气网眼纹理在交织变换。视频展示了节奏紧凑、充满力量感的网页向下滚动效果，以及鼠标悬停时强烈的视觉放大与颜色反转等 UI 交互动作。"
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124, num_inference_steps=50, seed=42,
    references=[{"type": "image", "image": Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/0.png").convert("RGB")}]
)
write_video_audio(
    video=video, audio=audio,
    output_path="ti2va.mp4", fps=24, audio_sample_rate=32000,
)

# Text + Reference Audio + Reference Video -> Video + Audio
ref_video, ref_video_audio, ref_video_sample_rate = read_video_audio(
    "data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video.mp4",
    height=480, width=832, num_frames=124, fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
ref_voice, ref_voice_sample_rate = read_audio(
    "data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/voice.mp3", duration=len(ref_video) / 24, resample=True, resample_rate=pipe.audio_vae.sample_rate,
)
prompt = "subject_definitions:\n<Subject 1> is the young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, an unbuttoned white shirt, and silver rings, holding a small black lamb in his arms in <Video 1>.\n<Video 1> is the source video for the editing task.\n<Audio 1> is the synchronized audio track of <Video 1>, providing the background music.\n<Audio 2> is the voice timbre reference for <Subject 1>'s voice, containing a spoken male voiceover.\n\nsummary:\n[video editing + audio reference + audio reuse] The target video is an edited version of <Video 1>. <Subject 1>, wearing a bright pink suit and holding a black lamb, stands in a grassy field with other white lambs in the background. The edit animates <Subject 1>'s face to speak the user-provided dialogue. <Audio 1> is partially reused as the continuous background music, while the target references the calm male voice timbre of <Audio 2> for <Subject 1>'s spoken lines.\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - the man retains his identity, wavy blonde hair, pink suit, white shirt, accessories, and the black lamb he holds, with his mouth newly animated to speak.\n<Video 1> (source video editing): fully_preserved - the original camera framing, warm golden hour lighting, grassy hill setting, and background white lambs are maintained while the central character is edited.\n<Audio 1>: partially_copy - the atmospheric background music from <Audio 1> is reused in the target video, mixed beneath the newly added spoken dialogue.\n<Audio 2>: reference - the target audio references the male voice timbre from <Audio 2> to generate <Subject 1>'s spoken dialogue.\n\ndetailed_description:\nThe target video is in realistic photographic style.\n[Shot 1] The shot begins from the source <Video 1>, showing <Subject 1>, a young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, and a casually unbuttoned white shirt. He stands confidently in a sunlit green pasture, gently holding a small black lamb securely in his arms. The warm, golden hour lighting casts soft shadows across his face and the bright pink fabric of his suit. Behind him, several white lambs stand and graze on the rolling grassy hill against a clear, pale blue sky. The atmospheric background music from <Audio 1> plays continuously throughout the scene. <Subject 1> physically speaks, his mouth movements naturally syncing to the new dialogue, with his voice timbre referencing the calm male delivery from <Audio 2>. Looking thoughtfully forward, <Subject 1> (S1) speaks softly, <d>[English] Follow the wind, live free.</d> As he delivers the line, he subtly shifts his weight, cradling the resting black lamb while the camera slowly pushes in. <Subject 1> (S1) continues his thought, <d>[English] Leave worries behind, enjoy the moment.</d> Exactly as his voice stops, his lips meet in a relaxed, peaceful smile, and his jaw ceases speaking motion. He then turns his gaze slightly away toward the horizon, gently stroking the black lamb's fleece with his fingers as the camera holds on this tranquil, sunlit state through the end of the video.\n\noverall_soundscape:\nThe soundscape consists of the continuous, atmospheric background music from <Audio 1>, overlaid with the clear, calm male dialogue spoken by the main character, referencing the voice timbre of <Audio 2>.\n\nnon_diegetic_music:\nThe atmospheric, sustained background music from <Audio 1> is reused as the continuous score, playing quietly beneath the spoken dialogue."
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124, num_inference_steps=50, seed=42,
    references=[
        {"type": "video_audio", "video": ref_video, "audio": ref_video_audio, "sample_rate": ref_video_sample_rate},
        {"type": "audio", "audio": ref_voice, "sample_rate": ref_voice_sample_rate},
    ],
)
write_video_audio(
    video=video, audio=audio,
    output_path="tav2va.mp4", fps=24, audio_sample_rate=32000,
)
