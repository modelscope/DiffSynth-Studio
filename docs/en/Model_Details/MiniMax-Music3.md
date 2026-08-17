# MiniMax-Music3

MiniMax-Music3 is a music generation model built on a two-stage cascade of an autoregressive language model and a flow-matching acoustic model. Given a music description and lyrics, it generates a stereo song with vocals.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [MiniMax/MiniMax-Music3](https://www.modelscope.cn/models/MiniMax/MiniMax-Music3) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 6GB VRAM.

```python
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
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[MiniMax/MiniMax-Music3](https://www.modelscope.cn/models/MiniMax/MiniMax-Music3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_music3/model_inference/MiniMax-Music3.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_music3/model_inference_low_vram/MiniMax-Music3.py)|—|—|—|—|

## Model Inference

The model is loaded via `MiniMaxMusic3Pipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `MiniMaxMusic3Pipeline` inference include:

* `prompt`: The music description, specifying genre, BPM, key, vocal characteristics and arrangement.
* `lyrics`: The lyrics. Structure tags such as `[verse]` and `[chorus]` must each be on their own line; text on the same line as a leading tag is dropped. Leave it empty to generate instrumental music.
* `max_audio_duration`: Upper bound on the generated audio length in seconds. The autoregressive stage may stop earlier, so the actual length can be shorter; the frame count is capped at 9000.
* `num_inference_steps`: Number of flow-matching steps per window.
* `cfg_scale`: Classifier-free guidance scale for the acoustic stage.
* `seed`: Random seed.
* `rand_device`: Device on which random numbers are drawn. Set it to `"cpu"` for results that reproduce independently of the compute device.
* `progress_bar_cmd`: Progress bar. One bar covering all steps is shown per window.

Generation proceeds in two stages: the autoregressive language model emits a semantic token and residual RVQ codes frame by frame, and its per-frame hidden states condition a chunked flow-matching model that produces Flow-VAE latents, which the vocoder finally synthesizes into a 44.1kHz stereo waveform. The discrete sampling in the autoregressive stage is sensitive to numerical precision, so the parameters of that stage stay resident in VRAM and layer-wise VRAM management applies to the vocoder only.

If you run out of VRAM, please refer to [VRAM Management](../Pipeline_Usage/Model_Inference.md#vram-management).

## Model Training

Training is not yet supported for MiniMax-Music3.
