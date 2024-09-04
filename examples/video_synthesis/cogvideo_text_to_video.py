from diffsynth import ModelManager, save_video, VideoData, download_models, CogVideoPipeline
from diffsynth.extensions.RIFE import RIFEInterpolater
import torch, os
os.environ["TOKENIZERS_PARALLELISM"] = "True"



def text_to_video(model_manager, prompt, seed, output_path):
    pipe = CogVideoPipeline.from_model_manager(model_manager)
    torch.manual_seed(seed)
    video = pipe(
        prompt=prompt,
        height=480, width=720,
        cfg_scale=7.0, num_inference_steps=200
    )
    save_video(video, output_path, fps=8, quality=5)


def edit_video(model_manager, prompt, seed, input_path, output_path):
    pipe = CogVideoPipeline.from_model_manager(model_manager)
    input_video = VideoData(video_file=input_path)
    torch.manual_seed(seed)
    video = pipe(
        prompt=prompt,
        height=480, width=720,
        cfg_scale=7.0, num_inference_steps=200,
        input_video=input_video, denoising_strength=0.7
    )
    save_video(video, output_path, fps=8, quality=5)


def self_upscale(model_manager, prompt, seed, input_path, output_path):
    pipe = CogVideoPipeline.from_model_manager(model_manager)
    input_video = VideoData(video_file=input_path, height=480*2, width=720*2).raw_data()
    torch.manual_seed(seed)
    video = pipe(
        prompt=prompt,
        height=480*2, width=720*2,
        cfg_scale=7.0, num_inference_steps=30,
        input_video=input_video, denoising_strength=0.4, tiled=True
    )
    save_video(video, output_path, fps=8, quality=7)


def interpolate_video(model_manager, input_path, output_path):
    rife = RIFEInterpolater.from_model_manager(model_manager)
    video = VideoData(video_file=input_path).raw_data()
    video = rife.interpolate(video, num_iter=2)
    save_video(video, output_path, fps=32, quality=5)



download_models(["CogVideoX-5B", "RIFE"])

model_manager = ModelManager(torch_dtype=torch.bfloat16)
model_manager.load_models([
    "models/CogVideo/CogVideoX-5b/text_encoder",
    "models/CogVideo/CogVideoX-5b/transformer",
    "models/CogVideo/CogVideoX-5b/vae/diffusion_pytorch_model.safetensors",
    "models/RIFE/flownet.pkl",
])

# Example 1
text_to_video(model_manager, "an astronaut riding a horse on Mars.", 0, "1_video_1.mp4")
edit_video(model_manager, "a white robot riding a horse on Mars.", 1, "1_video_1.mp4", "1_video_2.mp4")
self_upscale(model_manager, "a white robot riding a horse on Mars.", 2, "1_video_2.mp4", "1_video_3.mp4")
interpolate_video(model_manager, "1_video_3.mp4", "1_video_4.mp4")

# Example 2
text_to_video(model_manager, "a dog is running.", 1, "2_video_1.mp4")
edit_video(model_manager, "a dog with blue collar.", 2, "2_video_1.mp4", "2_video_2.mp4")
self_upscale(model_manager, "a dog with blue collar.", 3, "2_video_2.mp4", "2_video_3.mp4")
interpolate_video(model_manager, "2_video_3.mp4", "2_video_4.mp4")
