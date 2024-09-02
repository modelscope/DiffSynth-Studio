from diffsynth import ModelManager, save_video, VideoData, download_models, CogVideoPipeline
from diffsynth.extensions.RIFE import RIFEInterpolater
import torch, os
os.environ["TOKENIZERS_PARALLELISM"] = "True"



def generate_a_dog(model_manager):
    pipe = CogVideoPipeline.from_model_manager(model_manager)
    prompt = "a dog is running."
    torch.manual_seed(1)
    video = pipe(
        prompt=prompt,
        height=480, width=720,
        cfg_scale=7.0, num_inference_steps=200
    )
    save_video(video, "video_1.mp4", fps=8, quality=5)


def add_a_blue_collar(model_manager):
    pipe = CogVideoPipeline.from_model_manager(model_manager)
    prompt = "a dog with blue collar."
    input_video = VideoData(video_file="video_1.mp4")
    torch.manual_seed(2)
    video = pipe(
        prompt=prompt,
        height=480, width=720,
        cfg_scale=7.0, num_inference_steps=200,
        input_video=input_video, denoising_strength=0.7
    )
    save_video(video, "video_2.mp4", fps=8, quality=5)


def self_upscale(model_manager):
    pipe = CogVideoPipeline.from_model_manager(model_manager)
    prompt = "a dog with blue collar."
    input_video = VideoData(video_file="video_2.mp4", height=480*2, width=720*2).raw_data()
    torch.manual_seed(3)
    video = pipe(
        prompt=prompt,
        height=480*2, width=720*2,
        cfg_scale=7.0, num_inference_steps=30,
        input_video=input_video, denoising_strength=0.4, tiled=True
    )
    save_video(video, "video_3.mp4", fps=8, quality=7)


def interpolate_video(model_manager):
    rife = RIFEInterpolater.from_model_manager(model_manager)
    video = VideoData(video_file="video_3.mp4").raw_data()
    video = rife.interpolate(video, num_iter=2)
    save_video(video, "video_4.mp4", fps=32, quality=5)



download_models(["CogVideoX-5B", "RIFE"])

model_manager = ModelManager(torch_dtype=torch.bfloat16)
model_manager.load_models([
    "models/CogVideo/CogVideoX-5b/text_encoder",
    "models/CogVideo/CogVideoX-5b/transformer",
    "models/CogVideo/CogVideoX-5b/vae/diffusion_pytorch_model.safetensors",
    "models/RIFE/flownet.pkl",
])

generate_a_dog(model_manager)
add_a_blue_collar(model_manager)
self_upscale(model_manager)
interpolate_video(model_manager)
