import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from modelscope import dataset_snapshot_download
from PIL import Image

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
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Comfy-Org/MiniMax-H3", origin_file_pattern="diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

# Text -> Video + Audio
prompt = "A girl is very happy, she is speaking in english: \u201cI enjoy working with Diffsynth-Studio, it\u2019s a perfect framework.\u201d"
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124, num_inference_steps=50, seed=0,
)
write_video_audio(
    video=video, audio=audio,
    output_path="t2va.mp4", fps=24, audio_sample_rate=32000,
)

# Text + First Frame + Last Frame -> Video + Audio
dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Pruned-FL2VA/*")
first_frame = Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Pruned-FL2VA/first.png")
last_frame = Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Pruned-FL2VA/last.png")
prompt = "\u5ba4\u5185\u5bb6\u5ead\u4e89\u5435\u77ed\u5267\u573a\u666f\uff0c\u7ad6\u5c4f\u77ed\u5267\u8d28\u611f\uff0c\u771f\u5b9e\u771f\u4eba\u8868\u6f14\uff0c\u4e2d\u5f0f\u5bb6\u5ead/\u5c0f\u996d\u9986\u5ba4\u5185\u73af\u5883\uff0c\u6696\u8272\u706f\u5149\uff0c\u80cc\u666f\u6709\u7ea2\u8272\u88c5\u9970\u548c\u4e66\u6cd5\u5b57\u5e45\uff0c\u6d45\u666f\u6df1\uff0c\u60c5\u7eea\u5f3a\u70c8\uff0c\u526a\u8f91\u8282\u594f\u7d27\u51d1\u3002\u8868\u6f14\u8981\u6c42\uff1a\u771f\u5b9e\u77ed\u5267\u8868\u6f14\u98ce\u683c\uff0c\u4e0d\u8981\u5938\u5f20\u821e\u53f0\u8154\u3002\u7537\u4eba\u7684\u8bed\u6c14\u662f\u6124\u6012\u3001\u59d4\u5c48\u3001\u6025\u5207\u7684\u53cd\u9a73\uff0c\u4ed6\u8bf4\u201c\u4f60\u5230\u5e95\u60f3\u5e72\u4ec0\u4e48\uff1f\u201d\uff1b\u4e2d\u8001\u5e74\u5973\u6027\u7684\u8bed\u6c14\u662f\u5c16\u9510\u3001\u5f3a\u52bf\u3001\u5486\u5486\u903c\u4eba\u7684\u8d28\u95ee\uff0c\u5979\u8bf4\u201c\u4f60\u5fc5\u987b\u8d54\u94b1\uff01\u201d\u3002\u4e24\u4eba\u4e4b\u95f4\u6709\u5f3a\u70c8\u5bf9\u5ce8\u611f\uff0c\u8282\u594f\u9010\u6b65\u5347\u7ea7\u3002\u753b\u9762\u98ce\u683c\uff1a\u7ad6\u5c4f9:16\uff0c\u624b\u673a\u77ed\u5267\u8d28\u611f\uff0c\u771f\u4eba\u5b9e\u62cd\u611f\uff0c\u6d45\u666f\u6df1\uff0c\u5ba4\u5185\u6696\u5149\uff0c\u4e2d\u8fd1\u666f\u4e3a\u4e3b\uff0c\u9891\u7e41\u6b63\u53cd\u6253\u526a\u8f91\uff0c\u80cc\u666f\u4fdd\u6301\u751f\u6d3b\u5316\uff0c\u4e0d\u8981\u79d1\u5e7b\u3001\u4e0d\u8981\u53e4\u88c5\u3001\u4e0d\u8981\u52a8\u753b\u611f\u3002\u753b\u9762\u4e2d\u4e0d\u8981\u51fa\u73b0\u4efb\u4f55\u5b57\u5e55\u3001\u6587\u5b57\u3001\u5e73\u53f0\u6c34\u5370\u6216\u8d34\u7247\u3002 "
video, audio = pipe(
    prompt=prompt,
    height=832, width=480, num_frames=124, num_inference_steps=50, seed=0,
    keyframes=[first_frame, last_frame], keyframe_indices=[0, -1],
)
write_video_audio(
    video=video, audio=audio,
    output_path="fl2va.mp4", fps=24, audio_sample_rate=32000,
)
