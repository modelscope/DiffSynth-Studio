"""
Ace-Step 1.5 Base (non-turbo, 24 layers) — Text-to-Music inference example.

Uses cfg_scale=7.0 (standard CFG guidance) and more steps for higher quality.
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
            origin_file_pattern="acestep-v15-base/model.safetensors"
        ),
        ModelConfig(
            model_id="ACE-Step/Ace-Step1.5",
            origin_file_pattern="acestep-v15-base/model.safetensors"
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

prompt = "A cinematic orchestral piece with soaring strings and heroic brass"
lyrics = "[Intro - Orchestra]\n\n[Verse 1]\nAcross the mountains, through the valley\nA journey of a thousand miles\n\n[Chorus]\nRise above the stormy skies\nLet the music carry you"

audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    duration=30.0,
    seed=42,
    num_inference_steps=20,
    cfg_scale=7.0,  # Base model uses CFG
    shift=3.0,
)

sf.write("acestep-v15-base.wav", audio.cpu().numpy(), pipe.sample_rate)
print(f"Saved, shape: {audio.shape}")
