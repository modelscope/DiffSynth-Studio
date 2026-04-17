"""
Ace-Step 1.5 Turbo (shift=1) — Text-to-Music inference example.

Uses shift=1.0 (no timestep transformation) for smoother, slower denoising.
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

prompt = "A gentle acoustic guitar melody with soft piano accompaniment, peaceful and warm atmosphere"
lyrics = "[Verse 1]\nSunlight filtering through the trees\nA quiet moment, just the breeze\n\n[Chorus]\nPeaceful heart, open mind\nLeaving all the noise behind"

audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    duration=30.0,
    seed=42,
    num_inference_steps=8,
    cfg_scale=1.0,
    shift=1.0,  # shift=1: no timestep transformation
)

sf.write("acestep-v15-turbo-shift1.wav", audio.cpu().numpy(), pipe.sample_rate)
print(f"Saved, shape: {audio.shape}")
