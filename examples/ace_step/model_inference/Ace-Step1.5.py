"""
Ace-Step 1.5 — Text-to-Music (Turbo) inference example.

Demonstrates the standard text2music pipeline with structured parameters
(caption, lyrics, duration, etc.) — no LLM expansion needed.

For Simple Mode (LLM expands a short description), see:
    - Ace-Step1.5-SimpleMode.py
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

prompt = "An explosive, high-energy pop-rock track with a strong anime theme song feel. The song kicks off with a catchy, synthesized brass fanfare over a driving rock beat with punchy drums and a solid bassline."
lyrics = """[Intro - Synth Brass Fanfare]

[Verse 1]
黑夜里的风吹过耳畔
甜蜜时光转瞬即逝
脚步飘摇在星光上

[Chorus]
心电感应在震动间
拥抱未来勇敢冒险

[Outro - Instrumental]"""

audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    duration=30.0,
    seed=42,
    num_inference_steps=8,
    cfg_scale=1.0,
    shift=3.0,
)

sf.write("Ace-Step1.5.wav", audio.cpu().numpy(), pipe.sample_rate)
print(f"Saved to Ace-Step1.5.wav, shape: {audio.shape}, duration: {audio.shape[-1] / pipe.sample_rate:.1f}s")
