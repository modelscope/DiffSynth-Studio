"""
Ace-Step 1.5 — Text-to-Music with Simple Mode (LLM expansion).

Uses the ACE-Step LLM to expand a simple description into structured
parameters (caption, lyrics, bpm, keyscale, etc.), then feeds them
to the DiffSynth Pipeline.

The LLM expansion uses the target library's LLMHandler. If vLLM is
not available, it falls back to using pre-structured parameters.

Usage:
    python examples/ace_step/model_inference/Ace-Step1.5-SimpleMode.py
"""
import os
import sys
import json
import torch
import soundfile as sf

from diffsynth.pipelines.ace_step import AceStepPipeline, ModelConfig


# ---------------------------------------------------------------------------
# Simple Mode: LLM expansion
# ---------------------------------------------------------------------------

def try_load_llm_handler(checkpoint_dir: str, lm_model_path: str = "acestep-5Hz-lm-1.7B",
                         backend: str = "vllm"):
    """Try to load the target library's LLMHandler. Returns (handler, success)."""
    try:
        from acestep.llm_inference import LLMHandler
        handler = LLMHandler()
        status, success = handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=lm_model_path,
            backend=backend,
        )
        if success:
            print(f"[Simple Mode] LLM loaded via {backend} backend: {status}")
            return handler, True
        else:
            print(f"[Simple Mode] LLM init failed: {status}")
            return None, False
    except Exception as e:
        print(f"[Simple Mode] LLMHandler not available: {e}")
        return None, False


def expand_with_llm(llm_handler, description: str, duration: float = 30.0):
    """Expand a simple description using LLM Chain-of-Thought."""
    result = llm_handler.generate_with_stop_condition(
        caption=description,
        lyrics="",
        infer_type="dit",  # metadata only
        temperature=0.85,
        cfg_scale=1.0,
        use_cot_metas=True,
        use_cot_caption=True,
        use_cot_language=True,
        user_metadata={"duration": int(duration)},
    )

    if result.get("success") and result.get("metadata"):
        meta = result["metadata"]
        return {
            "caption": meta.get("caption", description),
            "lyrics": meta.get("lyrics", ""),
            "bpm": meta.get("bpm", 100),
            "keyscale": meta.get("keyscale", ""),
            "language": meta.get("language", "en"),
            "timesignature": meta.get("timesignature", "4"),
            "duration": meta.get("duration", duration),
        }

    print(f"[Simple Mode] LLM expansion failed: {result.get('error', 'unknown')}")
    return None


def fallback_expand(description: str, duration: float = 30.0):
    """Fallback: use description as caption with default parameters."""
    print(f"[Simple Mode] LLM not available. Using description as caption.")
    return {
        "caption": description,
        "lyrics": "",
        "bpm": 100,
        "keyscale": "",
        "language": "en",
        "timesignature": "4",
        "duration": duration,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Target library path (for LLMHandler)
    TARGET_LIB = os.path.join(os.path.dirname(__file__), "../../../../ACE-Step-1.5")
    if TARGET_LIB not in sys.path:
        sys.path.insert(0, TARGET_LIB)

    description = "a soft Bengali love song for a quiet evening"
    duration = 30.0

    # 1. Try to load LLM
    print("=" * 60)
    print("Ace-Step 1.5 — Simple Mode (LLM expansion)")
    print("=" * 60)
    print(f"\n[Simple Mode] Input: '{description}'")

    llm_handler, llm_ok = try_load_llm_handler(
        checkpoint_dir=TARGET_LIB,
        lm_model_path="acestep-5Hz-lm-1.7B",
    )

    # 2. Expand parameters
    if llm_ok:
        params = expand_with_llm(llm_handler, description, duration=duration)
        if params is None:
            params = fallback_expand(description, duration)
    else:
        params = fallback_expand(description, duration)

    print(f"\n[Simple Mode] Parameters:")
    print(f"  Caption: {params['caption'][:100]}...")
    print(f"  Lyrics: {len(params['lyrics'])} chars")
    print(f"  BPM: {params['bpm']}, Keyscale: {params['keyscale']}")
    print(f"  Language: {params['language']}, Time Sig: {params['timesignature']}")
    print(f"  Duration: {params['duration']}s")

    # 3. Load Pipeline
    print(f"\n[Pipeline] Loading Ace-Step 1.5 (turbo)...")
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

    # 4. Generate
    print(f"\n[Generation] Running Pipeline...")
    audio = pipe(
        prompt=params["caption"],
        lyrics=params["lyrics"],
        duration=params["duration"],
        seed=42,
        num_inference_steps=8,
        cfg_scale=1.0,
        shift=3.0,
    )

    output_path = "Ace-Step1.5-SimpleMode.wav"
    sf.write(output_path, audio.cpu().numpy(), pipe.sample_rate)
    print(f"\n[Done] Saved to {output_path}")
    print(f"  Shape: {audio.shape}, Duration: {audio.shape[-1] / pipe.sample_rate:.1f}s")


if __name__ == "__main__":
    main()
