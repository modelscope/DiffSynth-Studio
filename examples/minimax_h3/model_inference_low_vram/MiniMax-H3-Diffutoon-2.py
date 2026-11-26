import torch
from modelscope import dataset_snapshot_download
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data import VideoData, save_video
from diffsynth.core.quant import QuantizeConfig
from diffsynth.core.context_compress import KVCompressor
from diffsynth.utils.diffutoon import caption_windows, video_to_blur, video_to_lineart, generate_segments, process_long_video


# Load Diffutoon-2 pipeline
offload_vram_config = {
    "offload_dtype": "disk", "offload_device": "disk",
    "onload_dtype": "disk", "onload_device": "disk",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}
cpu_offload_vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}
quant_vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cuda",
    "onload_dtype": torch.bfloat16, "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    "quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw"),
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **offload_vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-fl2va-nf4.safetensors", **offload_vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **cpu_offload_vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **cpu_offload_vram_config),
        ModelConfig(model_id="DiffSynth-Studio/Diffutoon-2", origin_file_pattern="line-art_module.safetensors", **quant_vram_config),
        ModelConfig(model_id="DiffSynth-Studio/Diffutoon-2", origin_file_pattern="color_module.safetensors", **quant_vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 10, # Large memory required for cache
)
pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))

# Process input videos
dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir=".", allow_file_pattern="data/examples/diffutoon/*")
video_to_lineart("data/examples/diffutoon/input_video.mp4", "outline_video.mp4")
video_to_blur("data/examples/diffutoon/input_video.mp4", "color_video.mp4")
outline_video = VideoData("outline_video.mp4", height=992, width=992).raw_data()
color_video = VideoData("color_video.mp4", height=992, width=992).raw_data()
segments = generate_segments(len(outline_video), clip_length=124) # Use a short clip_length to reduce VRAM usage.

# In this example, we use a static caption. For a long video, please use a vllm to generate captions for each clips.
# caption_windows("data/examples/diffutoon/input_video.mp4", "captions.json", segments=[(0, 328), (323, 447)])
caption = "A young woman with long dark hair, in a white sailor-style dress with blue trim, a blue bow, and white platform shoes, dances through a series of poses in a bright room with floor-to-ceiling windows, a reflective floor, and a cityscape beyond. She begins standing with arms spread, then kicks one leg up while raising an arm overhead, balances on one leg with arms crossed, places hands on hips, makes a hand gesture near her face, and turns partly away with arms extended. She leans with a hand near her head, leaps into the air with both feet off the floor, and finishes balanced on one leg with the other extended and arms outstretched, her hair and dress moving with each shift."
prompt = f"Generate a Japanese anime-style cartoon video with clean line work and flat cel-shaded colors, following the outlines of the line-art video and the colors, lighting and motion of the blurred video.\ndetailed_description:\n{caption}"

# Generate video segments
controlnet_compress_config = KVCompressor([{"method": "entropack", "target_bpp": 4.0}])
video = process_long_video(pipe, outline_video, color_video, prompt, segments, seed=43, num_inference_steps=4, controlnet_compress_config=controlnet_compress_config)
save_video(video, "minimax_h3_diffutoon.mp4", fps=30, quality=5)
