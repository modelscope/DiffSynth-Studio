from diffsynth.pipelines.minimax_music3 import MiniMaxMusic3Pipeline, ModelConfig
import soundfile as sf
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = MiniMaxMusic3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="qwen_7B/qwen_7B/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="flowmatching_vae.pth", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="dav.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="qwen_7B/qwen3-8B-tokenizer-music/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

lyrics = """[verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[chorus]
Softly the world begins to breathe"""
prompt = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
    "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
    "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
)
audio = pipe(prompt=prompt, lyrics=lyrics, audio_duration=60.0, num_inference_steps=30, cfg_scale=1.7, seed=7)
sf.write("MiniMax-Music3.wav", audio.T, pipe.sample_rate)
