"""
Ace-Step 1.5 Turbo (shift=3) — Text-to-Music inference example.

Uses shift=3.0 (default turbo shift) for faster denoising convergence.
"""
from diffsynth.pipelines.ace_step import AceStepPipeline, ModelConfig
import torch
import soundfile as sf


pipe = AceStepPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="ACE-Step/Ace-Step1.5",
            origin_file_pattern="acestep-v15-turbo/model.safetensors"
        ),
        ModelConfig(
            model_id="ACE-Step/Ace-Step1.5",
            origin_file_pattern="acestep-v15-turbo/model.safetensors"
        ),
        ModelConfig(
            model_id="ACE-Step/Ace-Step1.5",
            origin_file_pattern="Qwen3-Embedding-0.6B/model.safetensors"
        ),
    ],
    tokenizer_config=ModelConfig(
        model_id="ACE-Step/Ace-Step1.5",
        origin_file_pattern="Qwen3-Embedding-0.6B/"
    ),
    vae_config=ModelConfig(
        model_id="ACE-Step/Ace-Step1.5",
        origin_file_pattern="vae/"
    ),
)

prompt = "An explosive, high-energy pop-rock track with anime theme song feel"
lyrics = "[Intro]\n\n[Verse 1]\nRunning through the neon lights\nChasing dreams across the night\n\n[Chorus]\nFeel the fire in my soul\nMusic takes complete control"

audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    duration=30.0,
    seed=42,
    num_inference_steps=8,
    cfg_scale=1.0,
    shift=3.0,
)

sf.write("acestep-v15-turbo-shift3.wav", audio.cpu().numpy(), pipe.sample_rate)
print(f"Saved, shape: {audio.shape}")
