from diffsynth.pipelines.yue2 import YuE2Pipeline, ModelConfig
from diffsynth.utils.data.audio import save_audio
import torch

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

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
cot = "full"
seed = 831001

pipe = YuE2Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="m-a-p/YuE2-3B", origin_file_pattern="model.safetensors", **vram_config),
        ModelConfig(model_id="m-a-p/YuE2-Vae", origin_file_pattern="model.safetensors", computation_dtype=torch.float32),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Plan the editable score, then revise it before rendering.
abc = pipe.plan(prompt=prompt, lyrics=lyrics, cot=cot, seed=seed)
with open("YuE2.abc", "w", encoding="utf-8") as f:
    f.write(abc)
abc = abc.replace('"C"', '"Cmaj7"').replace('"Em"', '"Em7"').replace('"Am"', '"Am7"').replace('"G"', '"G7"')

# Render the edited score.
audio = pipe(prompt=prompt, lyrics=lyrics, abc=abc, cot=cot, seed=seed)
save_audio(audio, 48000, "YuE2-Edit.wav")
