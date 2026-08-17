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

# `onload_device` must equal `computation_device` here: comfy-kitchen's `QuantizedTensor`
# cannot be deep-copied, and `AutoWrappedQuantizedModule.computation_module()` only skips
# its `copy.deepcopy` branch when the layer is already on the computation device.
vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Comfy-Org/MiniMax-H3", origin_file_pattern="diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors", **vram_config),
        ModelConfig(model_id="Comfy-Org/MiniMax-H3", origin_file_pattern="text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/processor/"),
)

# Text + Reference Image -> Video + Audio
dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Ref2VA/*")
ref_image = Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/0.png").convert("RGB")
prompt = "\u4e00\u4e2a\u7f51\u7ad9\u9875\u9762\uff0c\u7f51\u7ad9\u9875\u9762UI\u8bbe\u8ba1\uff0c\u7f51\u7ad9\u52a8\u6548\uff0c\u89c6\u9891\u5c55\u793a\u4e86\u6d41\u7545\u7684\u7f51\u9875\u5411\u4e0b\u6eda\u52a8\u6548\u679c\u3002\u4e00\u4e2a\u6781\u5177\u7206\u53d1\u529b\u4e0e\u52a8\u611f\u7684\u4ea7\u54c1\u5b98\u7f51\u98ce\u683c\u4ea7\u54c1\u843d\u5730\u9875 UI/UX \u6f14\u793a\u89c6\u9891\uff0c\u6838\u5fc3\u5c55\u793a\u4e3b\u4f53\u662f\u8be5\u4ea7\u54c1\u56fe\u72471\u3002\u9875\u9762\u91c7\u7528\u7c97\u72b7\u6709\u529b\u3001\u503e\u659c\u7684\u8d85\u5927\u53f7\u65e0\u886c\u7ebf\u5b57\u4f53\u8fdb\u884c\u5f20\u626c\u7684\u6392\u7248\u3002\u80cc\u666f\u6709\u6781\u5177\u901f\u5ea6\u611f\u7684\u52a8\u6001\u5149\u5f71\u3001\u6697\u8272\u78b3\u7ea4\u7ef4\u6216\u8fd0\u52a8\u900f\u6c14\u7f51\u773c\u7eb9\u7406\u5728\u4ea4\u7ec7\u53d8\u6362\u3002\u89c6\u9891\u5c55\u793a\u4e86\u8282\u594f\u7d27\u51d1\u3001\u5145\u6ee1\u529b\u91cf\u611f\u7684\u7f51\u9875\u5411\u4e0b\u6eda\u52a8\u6548\u679c\uff0c\u4ee5\u53ca\u9f20\u6807\u60ac\u505c\u65f6\u5f3a\u70c8\u7684\u89c6\u89c9\u653e\u5927\u4e0e\u989c\u8272\u53cd\u8f6c\u7b49 UI \u4ea4\u4e92\u52a8\u4f5c\u3002"
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
