import torch
from PIL import Image
import librosa
from diffsynth import save_video, VideoData, save_video_with_audio
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    audio_processor_config=ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/"),
)

prompt = "a person is singing"
input_image = Image.open("/mnt/nas1/zhanghong/project/aigc/Wan2.2_s2v/examples/pose.png").convert("RGB").resize((width, height))
# s2v audio input, recommend 16kHz sampling rate
audio_path = '/mnt/nas1/zhanghong/project/aigc/Wan2.2_s2v/examples/sing.MP3'
input_audio, sample_rate = librosa.load(audio_path, sr=16000)

# Speech-to-video
video = pipe(
    prompt=prompt,
    input_image=input_image,
    negative_prompt="",
    seed=0,
    num_frames=81,
    height=1280,
    width=720,
    audio_sample_rate=sample_rate,
    input_audio=input_audio,
    num_inference_steps=40,
)
save_video_with_audio(video, "video_with_audio.mp4", audio_path, fps=16, quality=5)
