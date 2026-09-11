from diffsynth.pipelines.yue2 import YuE2Pipeline, ModelConfig
from diffsynth.utils.data.audio import save_audio
import torch

pipe = YuE2Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="m-a-p/YuE2-3B", origin_file_pattern="model.safetensors"),
        ModelConfig(model_id="m-a-p/YuE2-Vae", origin_file_pattern="model.safetensors", computation_dtype=torch.float32),
    ],
)

prompt = "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"
lyrics = """[Verse]
Neon fades along the lane
Footsteps keep the time of rain
Fold the night and leave it here
Morning has a sky to clear

[Chorus]
Let the day come into view
Every road begins with you
Hold a little room for light
We will sing beyond the night"""

audio = pipe(prompt=prompt, lyrics=lyrics, cot="full", seed=831001)
save_audio(audio, 48000, "YuE2.wav")
