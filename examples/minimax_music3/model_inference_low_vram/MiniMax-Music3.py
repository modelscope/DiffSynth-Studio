from diffsynth.pipelines.minimax_music3 import MiniMaxMusic3Pipeline, ModelConfig
from diffsynth.utils.data.audio import save_audio
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
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="language_model/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="rvq_depth_decoder/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="condition_encoder/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="vocoder/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

lyrics = (
    "[verse]\n"
    "Morning light filtering through the pine\n"
    "Every quiet street is yours and mine\n"
    "[chorus]\n"
    "Softly the world begins to breathe"
)
prompt = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
    "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
    "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
)
audio = pipe(prompt=prompt, lyrics=lyrics, max_audio_duration=60.0, num_inference_steps=30, cfg_scale=1.7, seed=7)
save_audio(audio, 44100, "MiniMax-Music3.wav")
