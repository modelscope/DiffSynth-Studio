from diffsynth.pipelines.ace_step import AceStepPipeline, ModelConfig
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.utils.data.audio import save_audio, read_audio
from modelscope import snapshot_download
import torch

pipe = AceStepPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ACE-Step/acestep-v15-xl-sft", origin_file_pattern="model-*.safetensors"),
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/model.safetensors"),
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    text_tokenizer_config=ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/"),
)
pipe.load_lora(
    pipe.dit,
    # This LoRA is recommended.
    ModelConfig(model_id="DiffSynth-Studio/acestep15xlsft-lora-music", origin_file_pattern="model.safetensors")
)
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/acestep15xlsft-vocals2music")],
)

snapshot_download("DiffSynth-Studio/acestep15xlsft-vocals2music", allow_file_pattern="assets/vocals_male.wav", local_dir="data")
vocals, sample_rate = read_audio("data/assets/vocals_male.wav", resample=True, resample_rate=pipe.vae.sampling_rate)
lyrics = """
[Verse 1]
深夜的屏幕，微光在闪烁
指尖敲击着，未知的脉络
不再是孤岛，独自去摸索
这里有一片海，等待你停泊

从零到一的距离，不再遥远
丰富的模型，静候被点燃
打破围墙的界限，推倒高墙
让智慧的火花，自由地碰撞

[Pre-Chorus]
听，算法在呼吸，心跳同频共振
看，开放的火炬，照亮前行路程
每一个 Commit，都是真诚的见证
每一次 Fork，都连接着可能

[Chorus]
魔搭社区，汇聚世界的目光
开放的力量，让技术不再隐藏
我们在云端，编织梦想的网
探索的尽头，是无限的远方

ModelScope，连接你我心房
共享的代码，是最美的乐章
不论来自何方，无论身在何处
在这里创造，让未来发光

[Verse 2]
CV 的眼眸，看清世间万象
NLP 的低语，读懂文字芬芳
音频的波动，捕捉灵魂声响
多模态的世界，由此刻启航

不需要重复造轮子的疲惫
站在巨人的肩膀，看得更远
开发者的心血，开放的精神
在这里传承，变得如此纯粹

[Bridge]
也许会有疑惑，也许会有迷茫
但社区的温暖，是坚实的后盾墙
讨论区的热帖，指引方向
协作的光芒，比星光更亮

打破壁垒！
拥抱开放！
探索未知！
就在此刻！

[Chorus]
魔搭社区，汇聚世界的目光
开放的力量，让技术不再隐藏
我们在云端，编织梦想的网
探索的尽头，是无限的远方

让未来发光！
"""
prompt = "Music with clear vocals"
audio = template(
    pipe,
    prompt=prompt,
    lyrics=lyrics,
    duration=160,
    bpm=100,
    keyscale="B minor",
    timesignature="4",
    vocal_language="zh",
    seed=42,
    num_inference_steps=100,
    cfg_scale=1.0,
    template_inputs = [{"audio": (vocals, sample_rate), "scale": 1}],
    shift=6,
)
save_audio(audio, pipe.vae.sampling_rate, "audio_output.wav")