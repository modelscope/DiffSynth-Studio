# DiffSynth-Music

DiffSynth-Music is a controllable music generation model suite developed by the DiffSynth-Studio team, built upon ACE-Step-1.5. It retains the foundational architecture of ACE-Step-1.5 while incorporating several additional modules to support diverse control mechanisms, including beat, vocals, accompaniment, rhythm, and timbre references.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [DiffSynth-Studio/DiffSynth-Music](https://www.modelscope.cn/models/DiffSynth-Studio/DiffSynth-Music) model for inference. The example covers six use cases: native music synthesis, beats control, vocals control, accompaniment control, prosody control, and audio reference (timbre) control.

```python
import torch, torchaudio
from diffsynth.pipelines.diffsynth_music import DiffSynthMusicPipeline, ModelConfig
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.core.data.operators import LoadMultiTrackAudio
from diffsynth.utils.music_tools import extract_prosody, generate_click
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
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="template_prosody/"),
        ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="template_reference/"),
    ],
)

lyrics = "[Intro]\n\n你的歌词..."
prompt = "An explosive, high-energy pop-rock track with a strong anime theme song feel."
snapshot_download("DiffSynth-Studio/DiffSynth-Music", allow_file_pattern="assets/*", local_dir="data")

# Native Music Synthesis
audio = template(
    pipe,
    prompt=prompt, negative_prompt=pipe.default_negative_prompt,
    lyrics=lyrics,
    duration=240,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=50,
)
torchaudio.save("audio_1_output.mp3", audio, 48000)

# Beats Control
beats = generate_click(120, duration=240)
audio = template(
    pipe,
    prompt=prompt, negative_prompt=pipe.default_negative_prompt,
    lyrics=lyrics,
    duration=240,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"model_id": 0, "audio": beats}],
    negative_template_inputs=[{"model_id": 0, "audio": beats * 0}],
)
torchaudio.save("audio_2_output.mp3", audio, 48000)

# Vocals Control
audio = LoadMultiTrackAudio(division_factor=3840)("data/assets/audio_reference.mp3")
vocals = pipe.extract_track(audio, track="vocals")
audio = template(
    pipe,
    prompt=prompt, negative_prompt=pipe.default_negative_prompt,
    lyrics="",
    duration=vocals.shape[1] / 48000,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"model_id": 0, "audio": vocals}],
    negative_template_inputs=[{"model_id": 0, "audio": vocals}],
    target_audio=vocals, target_track="vocals",
)
torchaudio.save("audio_3_output.mp3", audio, 48000)

# Accompaniment Music Control
audio = LoadMultiTrackAudio(division_factor=3840)("data/assets/audio_reference.mp3")
music = pipe.extract_track(audio, track=["drums", "bass", "other"])
audio = template(
    pipe,
    prompt=prompt, negative_prompt=pipe.default_negative_prompt,
    lyrics=lyrics,
    duration=music.shape[1] / 48000,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"model_id": 0, "audio": music}],
    negative_template_inputs=[{"model_id": 0, "audio": music}],
    target_audio=music, target_track=["drums", "bass", "other"],
)
torchaudio.save("audio_4_output.mp3", audio, 48000)

# Prosody Control
audio = LoadMultiTrackAudio(division_factor=3840)("data/assets/audio_reference.mp3")
vocals = pipe.extract_track(audio, track="vocals")
prosody = extract_prosody(vocals)
audio = template(
    pipe,
    prompt=prompt, negative_prompt=pipe.default_negative_prompt,
    lyrics=lyrics,
    duration=prosody.shape[1] / 48000,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{"model_id": 1, "audio": prosody}],
    negative_template_inputs=[{"model_id": 1, "audio": prosody}],
)
torchaudio.save("audio_5_output.mp3", audio, 48000)

# Audio Reference
audio = LoadMultiTrackAudio(division_factor=3840)("data/assets/audio_reference.mp3")
audio = template(
    pipe,
    prompt="Music", negative_prompt="", # The timbre is controlled by the reference audio.
    lyrics=lyrics,
    duration=200,
    seed=42, tiled=True, cfg_scale=4, num_inference_steps=100,
    template_inputs=[{"model_id": 2, "audio": audio}],
)
torchaudio.save("audio_6_output.mp3", audio, 48000)
```

A low VRAM version is available at [examples/diffsynth_music/model_inference_low_vram/DiffSynth-Music.py](/examples/diffsynth_music/model_inference_low_vram/DiffSynth-Music.py).

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
| [DiffSynth-Studio/DiffSynth-Music](https://www.modelscope.cn/models/DiffSynth-Studio/DiffSynth-Music) | [code](/examples/diffsynth_music/model_inference/DiffSynth-Music.py) | [code](/examples/diffsynth_music/model_inference_low_vram/DiffSynth-Music.py) | [code](/examples/diffsynth_music/model_training/full/DiffSynth-Music.sh) | [code](/examples/diffsynth_music/model_training/validate_full/DiffSynth-Music.py) | - | - |

## Model Inference

The model is loaded via `DiffSynthMusicPipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `DiffSynthMusicPipeline` inference include:

* `prompt`: The music description, specifying genre, mood and style.
* `negative_prompt`: The negative prompt for classifier-free guidance.
* `lyrics`: The lyrics. Structure tags such as `[Intro]` and `[Verse]` must each be on their own line. Leave it empty to generate instrumental music.
* `bpm`: Beats per minute. Default: 100.
* `timesignature`: Time signature. Default: "4".
* `keyscale`: Key and scale. Default: "B minor".
* `input_audio`: Optional audio tensor used as the starting point, combined with `denoising_strength` for audio-to-audio generation.
* `duration`: Duration of the generated audio in seconds.
* `num_inference_steps`: Number of flow-matching steps.
* `cfg_scale`: Classifier-free guidance scale.
* `seed`: Random seed.
* `tiled`: Whether to use tiled VAE decoding.
* `kv_cache`: Pre-computed KV cache for faster inference.
* `target_audio` / `target_track`: Optional track fusion, replacing the given track of the output with the corresponding track of `target_audio`.

The `TemplatePipeline` provides three control adapters loaded via `template_inputs` with `model_id`:

* `model_id: 0` — template_control: beats / vocals / accompaniment control. The generated audio follows the rhythm of the given audio. Combined with `target_audio`, the target track is fused into the output.
* `model_id: 1` — template_prosody: the generated vocals follow the pitch and rhythm (prosody) of the given audio. Use `extract_prosody` to build the input.
* `model_id: 2` — template_reference: the timbre of the output is controlled by the reference audio.

If you run out of VRAM, please refer to [VRAM Management](../Pipeline_Usage/Model_Inference.md#vram-management).

## Model Training

The model supports LoRA training and full training of the DiT (trainable models: `dit`) on `(prompt, lyrics, audio)` data. See [examples/diffsynth_music/model_training](/examples/diffsynth_music/model_training/) for training scripts.
