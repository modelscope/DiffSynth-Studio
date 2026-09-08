import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.audio import save_audio

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-audio-vae-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="model_patches/ltx-2.5-duration-head-bf16.safetensors", **vram_config),
    ],
    load_duration_head=True,
)

prompt = "A girl is very happy, she is speaking: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
negative_prompt = pipe.default_negative_prompt["LTX-2.5"]
_, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=43,
    num_frames=121,
    frame_rate=24,
    num_inference_steps=30,
    generate_video=False,
)
save_audio(audio, pipe.audio_vocoder.output_sampling_rate, "ltx2.5_t2a.wav")
