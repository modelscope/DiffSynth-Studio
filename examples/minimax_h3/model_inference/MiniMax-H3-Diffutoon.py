import json
import os
import subprocess
import sys
import torch
from PIL import Image
from modelscope import dataset_snapshot_download
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data import LowMemoryVideo, VideoData
from diffsynth.utils.data.audio_video import read_video_audio, write_video_audio

max_pixels, num_frames, context_frames, fps, steps, seed = 1032192, 328, 5, 24, 4, 43
blur_sigma, control_scale = 15, [0.4, 0.4]
style_prefix = "Generate a Japanese anime-style cartoon video with clean line work and flat cel-shaded colors, following the outlines of the line-art video and the colors, lighting and motion of the blurred video."

vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/Diffutoon-2", origin_file_pattern="line-art_module.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/Diffutoon-2", origin_file_pattern="color_module.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 24,
)
pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))

dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir=".", allow_file_pattern="data/examples/diffutoon/*")
input_video, work_dir = "data/examples/diffutoon/input_video.mp4", "data/examples/diffutoon/input_video_auto"
os.makedirs(work_dir, exist_ok=True)
lineart_path, blur_path = f"{work_dir}/lineart.mp4", f"{work_dir}/blur{blur_sigma}.mp4"
if not os.path.exists(lineart_path):
    subprocess.run([sys.executable, "-m", "diffsynth.utils.controlnet.video_lineart", input_video, lineart_path], check=True)
if not os.path.exists(blur_path):
    subprocess.run([sys.executable, "-m", "diffsynth.utils.controlnet.video_blur", input_video, blur_path, str(blur_sigma)], check=True)

# Resolution: largest 32-aligned size under max_pixels, same aspect ratio as the source
src_h, src_w = VideoData(input_video).shape()
scale = (max_pixels / (src_w * src_h)) ** 0.5
h0, w0 = int(src_h * scale) // 32 * 32, int(src_w * scale) // 32 * 32
height, width = min((abs(w / h - src_w / src_h), h, w) for h in (h0, h0 + 32) for w in (w0, w0 + 32) if h > 0 and w > 0 and w * h <= max_pixels)[1:]

# Window schedule at 24fps (17n+5), stride = num_frames - context_frames
v = LowMemoryVideo(input_video)
src_fps = float(v.reader.get_meta_data()["fps"])
total_frames = 0
while int(round(total_frames * src_fps / fps)) < len(v):
    total_frames += 1
stride, segments, g = num_frames - context_frames, [], 0
while True:
    rem = total_frames - g
    if rem <= num_frames:
        segments.append((g, 17 * max(1, -(-(rem - 5) // 17)) + 5))
        break
    segments.append((g, num_frames))
    g += stride
control_frames = segments[-1][0] + segments[-1][1]

cap_json = f"{work_dir}/prompt_segments_w{num_frames}_c{context_frames}g.json"
g0s = [g0 for g0, _ in segments]
if not os.path.exists(cap_json):
    subprocess.run([sys.executable, "-m", "diffsynth.utils.controlnet.video_caption", input_video, cap_json, "--g0", ",".join(map(str, g0s)), "--window", str(num_frames), "--context", str(context_frames)], check=True)
caps = {int(k): v for k, v in json.load(open(cap_json))["captions"].items()}
prompts = [f"{style_prefix}\ndetailed_description:\n{caps[g0]}" for g0, _ in segments]

def load_control(path):
    frames, _, _ = read_video_audio(path, height=height, width=width, num_frames=control_frames, fps=fps)
    return list(frames) + [frames[-1]] * (control_frames - len(frames))

control_line, control_color = load_control(lineart_path), load_control(blur_path)

all_frames, audio_chunks, pin_latents = [], [], None
audio_sr = pipe.audio_vae.sample_rate
for k, (g0, win) in enumerate(segments):
    video, audio, latents = pipe(
        prompt=prompts[k], height=height, width=width, num_frames=win, num_inference_steps=steps, seed=seed + k,
        control_video=[control_line[g0:g0 + win], control_color[g0:g0 + win]], control_scale=control_scale,
        pin_latents=pin_latents, return_latents=True,
    )
    pin_latents = latents[:, :, -2:].detach()
    overlap = 0 if k == 0 else context_frames
    if overlap:
        base = len(all_frames) - overlap
        for i in range(overlap):
            all_frames[base + i] = Image.blend(all_frames[base + i].convert("RGB"), video[i].convert("RGB"), i / max(overlap - 1, 1))
    all_frames.extend(f.convert("RGB") for f in video[overlap:])
    if audio is not None:
        audio_chunks.append(audio[:, int(overlap / fps * audio_sr):])

all_frames = all_frames[:total_frames]
full_audio = torch.cat(audio_chunks, dim=-1)[:, : int(len(all_frames) / fps * audio_sr)] if audio_chunks else None
write_video_audio(video=all_frames, audio=full_audio, output_path="minimax_h3_diffutoon.mp4", fps=fps, audio_sample_rate=audio_sr)
