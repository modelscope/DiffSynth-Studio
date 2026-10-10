import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffsynth.core.quant import QuantizeConfig
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from modelscope import dataset_snapshot_download
from PIL import Image

# FP8 quantization requires a GPU with native FP8 support (sm90+).
fp8 = QuantizeConfig(method="minimax_h3_vdn_fp8")
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
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", quantize=fp8, **vram_config),
        ModelConfig(model_id="OpenVDN/vdn-minimax-h3", origin_file_pattern="stage-b-step-2000/linear_branch/model.safetensors", quantize=fp8, **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
pipe.load_lora(pipe.dit, ModelConfig(model_id="OpenVDN/vdn-minimax-h3", origin_file_pattern="stage-b-step-2000/adapters/default/adapter_model.safetensors"))

# Text -> Video + Audio
prompt = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a bright medium shot frames a cheerful young woman with shoulder-length dark hair sitting at a tidy wooden desk in a sunlit study, a laptop open in front of her and a steaming mug of tea beside it. She scrolls and types with an easy rhythm, then pauses as her expression brightens into a delighted smile. The camera pushes in with small amplitude at slow speed while the young woman with a warm, clear voice (S1) looks up toward the camera and says with a happy laugh: <d>[English] I enjoy working with Diffsynth-Studio, it's a perfect framework.</d> [Shot 2] At 00:09.000, the camera cuts to a close-up of her hand setting the mug down beside the softly glowing laptop screen while her laughter carries over from the previous shot, and she leans back in her chair with a satisfied smile as the afternoon light drifts slowly across the desk.

overall_soundscape: Quiet room ambience with soft keyboard clatter and a faint laptop fan hum. The mug taps lightly against the desk, followed by her cheerful laughter and the soft creak of the chair.

non_diegetic_music: A gentle acoustic-guitar melody at a moderate tempo with light shaker percussion, resolving into a warm sustained chord as the video ends."""
video, audio = pipe(
    prompt=prompt,
    height=768, width=1344, num_frames=345, num_inference_steps=50, seed=0,
)
write_video_audio(
    video=video, audio=audio,
    output_path="vdn-fp8-t2va.mp4", fps=24, audio_sample_rate=32000,
)

# Text + First Frame + Last Frame -> Video + Audio
dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-FL2VA/*")
first_frame = Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/first.png")
last_frame = Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/last.png")
prompt = """How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, vertical mobile short-drama look, a medium close-up begins in the framing established by Picture 1: a young man in a dark jacket stands face to face with a middle-aged woman inside a warmly lit Chinese home restaurant, red decorations and a framed calligraphy scroll on the wall behind him, shallow depth of field. The camera holds a static shot as he tightens his jaw, his eyes reddening with a mix of anger and grievance, and the young man with an angry, aggrieved voice (S1) protests: <d>[Chinese] 你到底想干什么？</d> [Shot 2] At 00:07.500, the shot cuts to the reverse angle, a medium close-up of the middle-aged woman in a purple turtleneck and plaid coat with a younger woman standing out of focus behind her, while the man's final words carry over from the previous shot. She widens her eyes, leans forward, and points at him as the middle-aged woman with a sharp, forceful voice (S2) demands: <d>[Chinese] 你必须赔钱！</d> Her arm stays raised as the two hold their tense standoff, settling into the staging, pose, and composition established by Picture 2 at the end of the video.

overall_soundscape: Indoor restaurant ambience with a low refrigerator hum, dishware clinking in a back kitchen, and tense footsteps on tile. Sharp breaths and the rustle of clothing punctuate the confrontation.

non_diegetic_music: A sparse low-string drone at a slow tempo with occasional muffled percussion hits, rising in volume through the confrontation before cutting off sharply at the end."""
video, audio = pipe(
    prompt=prompt,
    height=1344, width=768, num_frames=345, num_inference_steps=50, seed=0,
    keyframes=[first_frame, last_frame], keyframe_indices=[0, -1],
)
write_video_audio(
    video=video, audio=audio,
    output_path="vdn-fp8-fl2va.mp4", fps=24, audio_sample_rate=32000,
)
