"""
Ace-Step 1.5 SFT (supervised fine-tuned, 24 layers) — Text-to-Music inference example.

SFT variant is fine-tuned for specific music styles.
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
            origin_file_pattern="acestep-v15-sft/model.safetensors"
        ),
        ModelConfig(
            model_id="ACE-Step/Ace-Step1.5",
            origin_file_pattern="acestep-v15-sft/model.safetensors"
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

prompt = "A jazzy lo-fi beat with smooth saxophone and vinyl crackle, late night vibes"
lyrics = "[Intro - Vinyl crackle]\n\n[Verse 1]\nMidnight city, neon glow\nSmooth jazz flowing to and fro\n\n[Chorus]\nLay back, let the music play\nJazzy nights, dreams drift away"

audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    duration=30.0,
    seed=42,
    num_inference_steps=20,
    cfg_scale=7.0,
    shift=3.0,
)

sf.write("acestep-v15-sft.wav", audio.cpu().numpy(), pipe.sample_rate)
print(f"Saved, shape: {audio.shape}")
