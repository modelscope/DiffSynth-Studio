import torch
from diffsynth.core.quant import QuantizeConfig
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

# fp8 quantizes the wide Linears while each model loads (the branch carries one of them).
# The quantized layers stay VRAM-managed, so the LoRA below hot-loads on top instead of
# being folded, and the whole pipeline runs on the regular offload configuration.
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
# Stage B trained this LoRA jointly with the linear branch; base + branch alone is not stage-b.
pipe.load_lora(pipe.dit, ModelConfig(model_id="OpenVDN/vdn-minimax-h3", origin_file_pattern="stage-b-step-2000/adapters/default/adapter_model.safetensors"))

# Text -> Video + Audio
prompt = "A girl is very happy, she is speaking in english: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
video, audio = pipe(
    prompt=prompt,
    height=768, width=1344, num_frames=345, num_inference_steps=50, seed=0,
)
write_video_audio(
    video=video, audio=audio,
    output_path="vdn-fp8-t2va.mp4", fps=24, audio_sample_rate=32000,
)