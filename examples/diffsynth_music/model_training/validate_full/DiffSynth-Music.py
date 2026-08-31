import torch, torchaudio
from diffsynth.pipelines.diffsynth_music import DiffSynthMusicPipeline, ModelConfig
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.core.data.operators import LoadMultiTrackAudio
from diffsynth.utils.music_tools import extract_prosody, generate_click
from diffsynth import load_state_dict
from modelscope import snapshot_download


pipe = DiffSynthMusicPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="transformer/model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="conditioner/model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="vae/model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="track_separator/model.safetensors", computation_dtype=torch.float32),
    ],
    tokenizer_config=ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="text_encoder/"),
)
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="template_control/"),
    ],
)
template.models[0].load_state_dict(load_state_dict("./models/train/DiffSynth-Music_full/epoch-1.safetensors", torch_dtype=torch.bfloat16, device="cuda"))

lyrics = "[Intro]\n\n清新海风里有我们旅途\n漆黑海浪上有帆依呀远征\n风暴的咆哮不把恐惧藏水手的胸襟\n祈祷你像无畏的领航人\n懂也不懂的守护航程\n你在甲板上留下的刻痕\n是我梦的风景\n\n我要送你永不沉的信念\n升起代表勇的黑旗幡\n我要送你永不沉的誓言\n锚连着锚把七海踏遍\n你就是烈焰\n你就是烈焰\n我的血未寒\n不灭的烽火燃在你身边\n我的血未寒\n\n怒海的狂涛总是起了又平\n凝望指着罗盘的星辰\n我要把酒全都灌进骨里\n陪我一起远行\n\n我要送你永不沉的信念\n升起代表勇的黑旗幡\n我要送你永不沉的誓言\n锚连着锚把七海踏遍\n你就是烈焰\n你就是烈焰\n我的血未寒\n不灭的烽火燃在你身边\n我的血未寒\n\n祈祷你像无畏的领航人\n懂也不懂的守护航程\n你在甲板上留下的刻痕\n是我梦的风景\n\n我要送你永不沉的信念\n升起代表勇的黑旗幡\n我要送你永不沉的誓言\n锚连着锚把七海踏遍\n你就是烈焰\n你就是烈焰\n我的血未寒\n不灭的烽火燃在你身边\n我的血未寒\n\n我要送你永不沉的信念\n升起代表勇的黑旗幡\n我要送你永不沉的誓言\n锚连着锚把七海踏遍\n你就是烈焰\n你就是烈焰\n我的血未寒\n不灭的烽火燃在你身边\n我的血未寒\n"
prompt = "An explosive, high-energy pop-rock track with a strong anime theme song feel."
snapshot_download("DiffSynth-Studio/DiffSynth-Music", allow_file_pattern="assets/*", local_dir="data")

# Beats Control
bpm = 120
duration = 240
beats = generate_click(bpm, duration=duration)
torchaudio.save("audio_2_input.mp3", beats, 48000)
audio = template(
    pipe,
    prompt=prompt, negative_prompt=pipe.default_negative_prompt,
    lyrics=lyrics,
    duration=duration,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"model_id": 0, "audio": beats}],
    negative_template_inputs=[{"model_id": 0, "audio": beats * 0}],
)
torchaudio.save("audio_2_output.mp3", audio, 48000)
torchaudio.save("audio_2_output_with_beats.mp3", audio + beats, 48000)
