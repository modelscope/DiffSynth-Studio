"""
Ace-Step 1.5 XL Turbo (32 layers) — Text-to-Music inference example.

XL turbo with fast generation (8 steps, shift=3.0, no CFG).
"""
from diffsynth.pipelines.ace_step import AceStepPipeline, ModelConfig
import torch
import soundfile as sf


pipe = AceStepPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="ACE-Step/acestep-v15-xl-turbo",
            origin_file_pattern="model-*.safetensors"
        ),
        ModelConfig(
            model_id="ACE-Step/acestep-v15-xl-turbo",
            origin_file_pattern="model-*.safetensors"
        ),
        ModelConfig(
            model_id="ACE-Step/acestep-v15-xl-turbo",
            origin_file_pattern="Qwen3-Embedding-0.6B/model.safetensors"
        ),
    ],
    tokenizer_config=ModelConfig(
        model_id="ACE-Step/acestep-v15-xl-turbo",
        origin_file_pattern="Qwen3-Embedding-0.6B/"
    ),
    vae_config=ModelConfig(
        model_id="ACE-Step/acestep-v15-xl-turbo",
        origin_file_pattern="vae/"
    ),
)

prompt = "An upbeat electronic dance track with pulsing synths and driving bassline"
lyrics = "[Intro - Synth build]\n\n[Verse 1]\nFeel the rhythm in the air\nElectric beats are everywhere\n\n[Drop]\n\n[Chorus]\nDance until the break of dawn\nMove your body, carry on"

audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    duration=30.0,
    seed=42,
    num_inference_steps=8,
    cfg_scale=1.0,  # turbo: no CFG
    shift=3.0,
)

sf.write("acestep-v15-xl-turbo.wav", audio.cpu().numpy(), pipe.sample_rate)
print(f"Saved, shape: {audio.shape}")
