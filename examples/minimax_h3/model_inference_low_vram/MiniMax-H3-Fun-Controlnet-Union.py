import torch
from PIL import Image, ImageDraw
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data import VideoData
from diffsynth.utils.data.audio_video import write_video_audio
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
# The ControlNet's state dict converter fuses `to_q` / `to_k` / `to_v` into one `qkv_proj`, and disk
# offload resolves exactly one source tensor per parameter with no transform, so the branch is kept
# on CPU offload instead. It is 6.8 GB against the 62 GB transformer, so little is lost.
controlnet_vram_config = {**vram_config, "offload_dtype": torch.bfloat16, "offload_device": "cpu", "onload_dtype": torch.bfloat16, "onload_device": "cpu"}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
        ModelConfig(model_id="PAI/MiniMax-H3-Fun-Controlnet-Union", origin_file_pattern="MiniMax-H3-Fun-Controlnet-Union.safetensors", **controlnet_vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)

dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Fun-Controlnet-Union/*")
dataset_base_path = "data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union"
height, width, num_frames = 480, 832, 124
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# Control video (canny / depth / hed / mlsd / pose) -> Video + Audio.
# The checkpoint is guidance-distilled, so cfg_scale stays at 1. `control_scale` weights every
# control skip before it is added to the main branch: 1.0 is the strongest control, 0.0 disables it.
control_video = VideoData(f"{dataset_base_path}/control_video.mp4", height=height, width=width)
control_video = [control_video[i] for i in range(num_frames)]
prompt = "A young woman with long wavy red hair gently turns her head to the right, offering a soft greeting to someone off-camera. She wears an olive-green knit poncho, and the soft diffused light of dusk falls across her face. Distant building silhouettes stand against a cloudy sky."
video, audio = pipe(
    prompt=prompt, negative_prompt=negative_prompt,
    height=height, width=width, num_frames=num_frames,
    num_inference_steps=40, seed=43,
    control_video=control_video, control_scale=1.0,
)
write_video_audio(
    video=video, audio=audio, output_path="minimax_h3_fun_controlnet_union_control.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)

# Control video + Inpaint -> Video + Audio. The released checkpoint carries the inpaint channels
# (control_in_dim=49), so the same weights take a source video plus a mask marking what to
# regenerate: white repaints, black keeps the `inpaint_video` content. A large static mask tends to
# suppress the generated soundtrack; keep the mask to the region that actually needs repainting when
# the audio matters.
inpaint_video = VideoData(f"{dataset_base_path}/video.mp4", height=height, width=width)
inpaint_video = [inpaint_video[i] for i in range(num_frames)]
mask_frame = Image.new("L", (width, height), 0)
ImageDraw.Draw(mask_frame).rectangle([width // 4, height // 4, width * 3 // 4, height * 3 // 4], fill=255)
inpaint_video_mask = [mask_frame] * num_frames
video, audio = pipe(
    prompt=prompt, negative_prompt=negative_prompt,
    height=height, width=width, num_frames=num_frames,
    num_inference_steps=40, seed=43,
    control_video=control_video, control_scale=1.0,
    inpaint_video=inpaint_video, inpaint_video_mask=inpaint_video_mask,
)
write_video_audio(
    video=video, audio=audio, output_path="minimax_h3_fun_controlnet_union_inpaint.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
