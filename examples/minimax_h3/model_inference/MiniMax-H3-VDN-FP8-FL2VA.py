import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

# fp8 replaces AutoWrappedLinear with a plain quantized Linear that VRAM management can no
# longer route, so the DiT stays resident and only the encoders/decoders offload. The fp8
# DiT is ~33 GB; the bf16 peak during the swap is ~70 GB, which fits an 80 GB card.
resident = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cuda",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
offload = {**resident, "offload_device": "cpu", "onload_device": "cpu"}

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **offload),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **resident),
        ModelConfig(model_id="OpenVDN/vdn-minimax-h3", origin_file_pattern="stage-b-step-2000/linear_branch/model.safetensors", **offload),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **offload),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **offload),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
# Stage B trained this LoRA jointly with the linear branch; base + branch alone is not stage-b.
# Must be folded BEFORE fp8: the quantiser reads Linear.weight once to build the fp8 copy.
pipe.load_lora(pipe.dit, ModelConfig(model_id="OpenVDN/vdn-minimax-h3", origin_file_pattern="stage-b-step-2000/adapters/default/adapter_model.safetensors"))

# Text -> Video + Audio
prompt = "A girl is very happy, she is speaking in english: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
video, audio = pipe(
    prompt=prompt,
    height=768, width=1344, num_frames=345, num_inference_steps=50, seed=0,
    # "auto" is upstream's default: decomposed on any CUDA device (the only backend that
    # scales past ~64k tokens without a BlockMask), flex without CUDA. Pin
    # "decomposed"/"flex"/"fa4" to force one, or "ref" for the bitwise eager oracle.
    vdn_softmax_impl="auto",
    use_fused_kernels=True,
    fp8=True,
)
write_video_audio(
    video=video, audio=audio,
    output_path="vdn-fp8-t2va.mp4", fps=24, audio_sample_rate=32000,
)
