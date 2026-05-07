# ACE-Step

ACE-Step 1.5 is an open-source music generation model based on DiT architecture, supporting text-to-music, audio cover, repainting and other functionalities, running efficiently on consumer-grade hardware.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [ACE-Step/Ace-Step1.5](https://www.modelscope.cn/models/ACE-Step/Ace-Step1.5) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 3GB VRAM.

```python
from diffsynth.pipelines.ace_step import AceStepPipeline, ModelConfig
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


pipe = AceStepPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="acestep-v15-turbo/model.safetensors", **vram_config),
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/model.safetensors", **vram_config),
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    text_tokenizer_config=ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = "An explosive, high-energy pop-rock track with a strong anime theme song feel. The song kicks off with a catchy, synthesized brass fanfare over a driving rock beat with punchy drums and a solid bassline. A powerful, clear male vocal enters with a theatrical and energetic delivery, soaring through the verses and hitting powerful high notes in the chorus. The arrangement is dense and dynamic, featuring rhythmic electric guitar chords, brief instrumental breaks with synth flourishes, and a consistent, danceable groove throughout. The overall mood is triumphant, adventurous, and exhilarating."
lyrics = '[Intro - Synth Brass Fanfare]\n\n[Verse 1]\n黑夜里的风吹过耳畔\n甜蜜时光转瞬即万\n脚步飘摇在星光上\n心追节奏心跳狂乱\n耳边传来电吉他呼唤\n手指轻触碰点流点燃\n梦在云端任它蔓延\n疯狂跳跃自由无间\n\n[Chorus]\n心电感应在震动间\n拥抱未来勇敢冒险\n那旋律在心中无限\n世界变得如此耀眼\n\n[Instrumental Break - Synth Brass Melody]\n\n[Verse 2]\n鼓点撞击黑夜的底端\n跳动节拍连接你我俩\n在这里让灵魂发光\n燃尽所有不留遗憾\n\n[Instrumental Break - Synth Brass Melody]\n\n[Bridge]\n光影交错彼此的视线\n霓虹之下夜空的蔚蓝\n月光洒下温热心田\n追逐梦想它不会遥远\n\n[Chorus]\n心电感应在震动间\n拥抱未来勇敢冒险\n那旋律在心中无限\n世界变得如此耀眼\n\n[Outro - Instrumental with Synth Brass Melody]\n[Song ends abruptly]'
audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    duration=160,
    bpm=100,
    keyscale="B minor",
    timesignature="4",
    vocal_language="zh",
    seed=42,
)

save_audio(audio, pipe.vae.sampling_rate, "acestep-v15-turbo.wav")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[ACE-Step/Ace-Step1.5](https://www.modelscope.cn/models/ACE-Step/Ace-Step1.5)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/Ace-Step1.5.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/Ace-Step1.5.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/Ace-Step1.5.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/Ace-Step1.5.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/Ace-Step1.5.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/Ace-Step1.5.py)|
|[ACE-Step/acestep-v15-turbo-shift1](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-shift1)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-turbo-shift1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-shift1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-turbo-shift1.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-turbo-shift1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-turbo-shift1.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-shift1.py)|
|[ACE-Step/acestep-v15-turbo-shift3](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-shift3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-turbo-shift3.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-shift3.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-turbo-shift3.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-turbo-shift3.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-turbo-shift3.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-shift3.py)|
|[ACE-Step/acestep-v15-turbo-continuous](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-continuous)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-turbo-continuous.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-continuous.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-turbo-continuous.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-turbo-continuous.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-turbo-continuous.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-continuous.py)|
|[ACE-Step/acestep-v15-base](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-base.py)|
|[ACE-Step/acestep-v15-base: CoverTask](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-base-CoverTask.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-base-CoverTask.py)|—|—|—|—|
|[ACE-Step/acestep-v15-base: RepaintTask](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-base-RepaintTask.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-base-RepaintTask.py)|—|—|—|—|
|[ACE-Step/acestep-v15-sft](https://www.modelscope.cn/models/ACE-Step/acestep-v15-sft)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-sft.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-sft.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-sft.py)|
|[ACE-Step/acestep-v15-xl-base](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-xl-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-xl-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-xl-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-xl-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-xl-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-xl-base.py)|
|[ACE-Step/acestep-v15-xl-sft](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-sft)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-xl-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-xl-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-xl-sft.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-xl-sft.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-xl-sft.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-xl-sft.py)|
|[ACE-Step/acestep-v15-xl-turbo](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-turbo)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference/acestep-v15-xl-turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_inference_low_vram/acestep-v15-xl-turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/full/acestep-v15-xl-turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_full/acestep-v15-xl-turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/lora/acestep-v15-xl-turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ace_step/model_training/validate_lora/acestep-v15-xl-turbo.py)|

## Model Inference

The model is loaded via `AceStepPipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `AceStepPipeline` inference include:

* `prompt`: Text description of the music.
* `cfg_scale`: Classifier-free guidance scale, defaults to 1.0.
* `lyrics`: Lyrics text.
* `task_type`: Task type,可选 values include `"text2music"` (text-to-music), `"cover"` (audio cover), `"repaint"` (repainting), defaults to `"text2music"`.
* `reference_audios`: List of reference audio tensors for timbre reference.
* `src_audio`: Source audio tensor for cover or repaint tasks.
* `denoising_strength`: Denoising strength, controlling how much the output is influenced by source audio, defaults to 1.0.
* `audio_cover_strength`: Audio cover step ratio, controlling how many steps use cover condition in cover tasks, defaults to 1.0.
* `audio_code_string`: Input audio code string for cover tasks with discrete audio codes.
* `repainting_ranges`: List of repainting time ranges (tuples of floats, in seconds) for repaint tasks.
* `repainting_strength`: Repainting intensity, controlling the degree of change in repainted areas, defaults to 1.0.
* `duration`: Audio duration in seconds, defaults to 60.
* `bpm`: Beats per minute, defaults to 100.
* `keyscale`: Musical key scale, defaults to "B minor".
* `timesignature`: Time signature, defaults to "4".
* `vocal_language`: Vocal language, defaults to "unknown".
* `seed`: Random seed.
* `rand_device`: Device for noise generation, defaults to "cpu".
* `num_inference_steps`: Number of inference steps, defaults to 8.
* `shift`: Timestep shift parameter for the scheduler, defaults to 3.0.

## Model Training

Models in the ace_step series are trained uniformly via `examples/ace_step/model_training/train.py`. The script parameters include:

* General Training Parameters
    * Dataset Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Path to the dataset metadata file.
        * `--dataset_repeat`: Number of dataset repeats per epoch.
        * `--dataset_num_workers`: Number of processes per DataLoader.
        * `--data_file_keys`: Field names to load from metadata, typically paths to image or video files, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths to load models from, in JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, separated by commas.
        * `--extra_inputs`: Additional input parameters required by the model Pipeline, separated by `,`.
        * `--fp8_models`: Models to load in FP8 format, currently only supported for models whose parameters are not updated by gradients.
    * Basic Training Configuration
        * `--learning_rate`: Learning rate.
        * `--num_epochs`: Number of epochs.
        * `--trainable_models`: Trainable models, e.g., `dit`, `vae`, `text_encoder`.
        * `--find_unused_parameters`: Whether unused parameters exist in DDP training.
        * `--weight_decay`: Weight decay magnitude.
        * `--task`: Training task, defaults to `sft`.
    * Output Configuration
        * `--output_path`: Path to save the model.
        * `--remove_prefix_in_ckpt`: Remove prefix in the model's state dict.
        * `--save_steps`: Interval in training steps to save the model.
    * LoRA Configuration
        * `--lora_base_model`: Which model to add LoRA to.
        * `--lora_target_modules`: Which layers to add LoRA to.
        * `--lora_rank`: Rank of LoRA.
        * `--lora_checkpoint`: Path to LoRA checkpoint.
        * `--preset_lora_path`: Path to preset LoRA checkpoint for LoRA differential training.
        * `--preset_lora_model`: Which model to integrate preset LoRA into, e.g., `dit`.
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.
        * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
    * Resolution Configuration
        * `--height`: Height of the image/video. Leave empty to enable dynamic resolution.
        * `--width`: Width of the image/video. Leave empty to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area, images larger than this will be scaled down during dynamic resolution.
        * `--num_frames`: Number of frames for video (video generation models only).
* ACE-Step Specific Parameters
    * `--tokenizer_path`: Tokenizer path, in format model_id:origin_pattern.
    * `--silence_latent_path`: Silence latent path, in format model_id:origin_pattern.
    * `--initialize_model_on_cpu`: Whether to initialize models on CPU.

### Example Dataset

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
