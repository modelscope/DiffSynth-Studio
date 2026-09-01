import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data import VideoData
from diffsynth.utils.data.audio_video import write_video_audio
from diffsynth import load_state_dict
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
        ModelConfig(model_id="PAI/MiniMax-H3-Fun-Controlnet-Union", origin_file_pattern="MiniMax-H3-Fun-Controlnet-Union.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
pipe.controlnet.load_state_dict(load_state_dict("./models/train/MiniMax-H3-Fun-Controlnet-Union_full/epoch-1.safetensors", torch_dtype=torch.bflaot16, device="cpu"), assign=True)
dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Fun-Controlnet-Union/*")

# Control video (canny / depth / hed / mlsd / pose) -> Video + Audio.
control_video = VideoData("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union/control_video.mp4", height=480, width=832)
control_video = [control_video[i] for i in range(124)]
prompt = "A T-Rex riding a bicycle, with explosions in the background."
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124,
    num_inference_steps=40, seed=43,
    control_video=control_video, control_scale=1.0,
)
write_video_audio(
    video=video, audio=audio, output_path="minimax_h3_fun_controlnet_union_control.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
