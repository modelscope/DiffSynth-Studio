import os
os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = '/root/models'
import torch
from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.models.minimax_h3_vace import MiniMaxH3VaceModel
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
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
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
# `vace_layers` must match the value used during training.
pipe.vace = MiniMaxH3VaceModel(
    vace_layers=(0, 8, 16, 24, 32, 40),
    hidden_size=pipe.dit.hidden_size,
    num_attention_heads=pipe.dit.num_attention_heads,
).to(dtype=pipe.torch_dtype, device=pipe.device)
state_dict = load_state_dict("/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/models/train/MiniMax-H3-VACE-full/step-50.safetensors")
pipe.vace.load_state_dict(state_dict)

lineart_video_path = "/mnt/nas3/sunyuzework/Diffutoon-2/data/xinhaicheng_39_lineart/1.mp4"

max_pixels, num_frames = 1044480, 39

vace_video = UnifiedDataset.default_video_operator(
    base_path="", max_pixels=max_pixels, height=None, width=None,
    height_division_factor=32, width_division_factor=32,
    num_frames=num_frames,
    time_division_factor=17, time_division_remainder=5,
    frame_rate=24, fix_frame_rate=True,
)(lineart_video_path)
width, height = vace_video[0].size
prompt = "subject_definitions:\n<Video 1> is the source line-art video that defines every visual element of the output: composition, character layout, poses, object placement, motion paths, and camera movement.\n\nsummary:\n[reference generation] The target video is a fully colored anime rendering of <Video 1>. This is a rendering task, not a generation task: the line art already specifies the complete content, and the output only adds color, shading, lighting, and material detail on top of it.\n\nretention_analysis:\n<Video 1> (composition, motion, camera, and all visual content): fully_preserved — every character position, pose, object, spatial layout, motion trajectory, and camera movement in the line art is retained exactly in the output.\n\ndetailed_description:\nRender exactly what <Video 1> shows, frame by frame. Every character, object, pose, background element, and camera movement must come from the line art and must match it exactly. Apply cel-shaded coloring, natural lighting, and material texture to the existing line work. Do not add, remove, move, or redesign anything that is not already present in the line art. Do not invent new objects, characters, backgrounds, or camera moves.\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A"

video, audio = pipe(
    prompt=prompt,
    height=height, width=width, num_frames=num_frames,
    num_inference_steps=50, seed=42,
    vace_video=vace_video,
)
write_video_audio(
    video=video, audio=audio, output_path="minimax_h3_vace_480p-1.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_vace_full.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
