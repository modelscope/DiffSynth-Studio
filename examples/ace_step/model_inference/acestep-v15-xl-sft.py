"""
Ace-Step 1.5 XL SFT (32 layers, supervised fine-tuned) — Text-to-Music inference example.
"""
from diffsynth.pipelines.ace_step import AceStepPipeline, ModelConfig
import torch
import soundfile as sf


pipe = AceStepPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="ACE-Step/acestep-v15-xl-sft",
            origin_file_pattern="model-*.safetensors"
        ),
        ModelConfig(
            model_id="ACE-Step/acestep-v15-xl-sft",
            origin_file_pattern="model-*.safetensors"
        ),
        ModelConfig(
            model_id="ACE-Step/acestep-v15-xl-sft",
            origin_file_pattern="Qwen3-Embedding-0.6B/model.safetensors"
        ),
    ],
    tokenizer_config=ModelConfig(
        model_id="ACE-Step/acestep-v15-xl-sft",
        origin_file_pattern="Qwen3-Embedding-0.6B/"
    ),
    vae_config=ModelConfig(
        model_id="ACE-Step/acestep-v15-xl-sft",
        origin_file_pattern="vae/"
    ),
)

prompt = "A beautiful piano ballad with lush strings and emotional vocals, cinematic feel"
lyrics = "[Intro - Solo piano]\n\n[Verse 1]\nWhispers of a distant shore\nMemories I hold so dear\n\n[Chorus]\nIn your eyes I see the dawn\nAll my fears are gone"

audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    duration=30.0,
    seed=42,
    num_inference_steps=20,
    cfg_scale=7.0,
    shift=3.0,
)

sf.write("acestep-v15-xl-sft.wav", audio.cpu().numpy(), pipe.sample_rate)
print(f"Saved, shape: {audio.shape}")
