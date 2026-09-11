# YuE2

YuE2 is a music generation model that unifies symbolic planning and audio generation. Given lyrics and a style prompt, it plans an editable melody-and-chord score, then realizes the plan as a complete 48 kHz stereo song with vocals and accompaniment.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e ".[yue2,audio]"
```

The `yue2` extra installs the `tiktoken` tokenizer dependency, and `audio` installs the audio utilities. For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [m-a-p/YuE2-3B](https://www.modelscope.cn/models/m-a-p/YuE2-3B) and [m-a-p/YuE2-Vae](https://www.modelscope.cn/models/m-a-p/YuE2-Vae) models for inference. VRAM management is enabled for the AR/NAR backbone, the framework automatically controls parameter loading based on available VRAM.

```python
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

pipe = YuE2Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="m-a-p/YuE2-3B", origin_file_pattern="model.safetensors", **vram_config),
        ModelConfig(model_id="m-a-p/YuE2-Vae", origin_file_pattern="model.safetensors", computation_dtype=torch.float32),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = (
    "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, "
    "lyrical memorable melody, unhurried phrasing, 88 BPM"
)
lyrics = (
    "[Verse]\n"
    "Neon fades along the lane\n"
    "Footsteps keep the time of rain\n"
    "[Chorus]\n"
    "Let the day come into view\n"
    "Every road begins with you"
)
audio = pipe(prompt=prompt, lyrics=lyrics, cot="full", seed=831001)
save_audio(audio, 48000, "YuE2.wav")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[m-a-p/YuE2-3B](https://www.modelscope.cn/models/m-a-p/YuE2-3B)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/yue2/model_inference/YuE2.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/yue2/model_inference_low_vram/YuE2.py)|—|—|—|—|
|[m-a-p/YuE2-3B: Edit](https://www.modelscope.cn/models/m-a-p/YuE2-3B)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/yue2/model_inference/YuE2-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/yue2/model_inference_low_vram/YuE2-Edit.py)|—|—|—|—|

## Model Inference

The model is loaded via `YuE2Pipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `YuE2Pipeline` inference include:

* `prompt`: Style description, including language, genre, instruments, vocal characteristics and tempo.
* `lyrics`: Lyrics text, with section tags such as `[Verse]` and `[Chorus]`.
* `cot`: Planning mode, optional values include `"full"` (melody with chord symbols), `"melody"` (melody only, recommended for covers) and `"off"` (no symbolic plan), defaults to `"full"`.
* `abc`: External ABC score used directly, requires `cot="full"` or `cot="melody"`.
* `cfg_scale`: Classifier-free guidance scale, defaults to 1.0 for `full`/`melody` and 1.01 for `off`.
* `seed`: Random seed.
* `num_inference_steps`: Number of flow-matching ODE steps for the acoustic stage, defaults to 30.
* `vae_core_frames`: Core frames per VAE decode tile, defaults to the value in the decoder configuration.
* `progress_bar_cmd`: Progress bar. The autoregressive stages show tokens/s and the acoustic stage shows steps/s.

Generation proceeds in three stages: the autoregressive score planner writes an editable ABC score, the autoregressive semantic generator emits discrete music tokens conditioned on the score, and the NAR flow-matching model synthesizes VAE latents from those tokens with a midpoint ODE, which the VAE finally decodes into a 48 kHz stereo waveform.

`YuE2Pipeline.plan` exports the symbolic score for editing. The plan-edit-render workflow plans a score, revises it (melody, chords, tempo or form), and renders the revised composition with `abc=...`; see [YuE2-Edit.py](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/yue2/model_inference/YuE2-Edit.py).

If you run out of VRAM, please refer to [VRAM Management](../Pipeline_Usage/Model_Inference.md#vram-management).

## Model Training

Training is not yet supported for YuE2.
