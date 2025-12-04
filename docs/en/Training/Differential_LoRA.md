# Differential LoRA Training

Differential LoRA training is a special form of LoRA training designed to enable models to learn differences between images.

## Training Approach

We were unable to identify the original proposer of differential LoRA training, as this technique has been circulating in the open-source community for a long time.

Assume we have two similar-content images: Image 1 and Image 2. For example, both images contain a car, but Image 1 has fewer details while Image 2 has more details. In differential LoRA training, we perform two-step training:

* Train LoRA 1 using Image 1 as training data with [standard supervised training](/docs/en/Training/Supervised_Fine_Tuning.md)
* Train LoRA 2 using Image 2 as training data, after integrating LoRA 1 into the base model, with [standard supervised training](/docs/en/Training/Supervised_Fine_Tuning.md)

In the first training step, since there is only one training image, the LoRA model easily overfits. Therefore, after training, LoRA 1 will cause the model to generate Image 1 without hesitation, regardless of the random seed. In the second training step, the LoRA model overfits again. Thus, after training, with the combined effect of LoRA 1 and LoRA 2, the model will generate Image 2 without hesitation. In short:

* LoRA 1 = Generate Image 1
* LoRA 1 + LoRA 2 = Generate Image 2

At this point, discarding LoRA 1 and using only LoRA 2, the model will understand the difference between Image 1 and Image 2, making the generated content tend toward "less like Image 1, more like Image 2."

Single training data can ensure the model overfits to the training data, but lacks stability. To improve stability, we can train with multiple image pairs and average the trained LoRA 2 models to obtain a more stable LoRA.

Using this training approach, some functionally unique LoRA models can be trained. For example, using ugly and beautiful image pairs to train LoRAs that enhance image aesthetics; using low-detail and high-detail image pairs to train LoRAs that increase image detail.

## Model Effects

We have trained several aesthetic enhancement LoRAs using differential LoRA training techniques. You can visit the corresponding model pages to view the generation effects.

* [DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1)
* [DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1](https://modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1)

## Using Differential LoRA Training in the Training Framework

The first step of training is identical to ordinary LoRA training. In the second step's training command, fill in the path of the first step's LoRA model file through the `--preset_lora_path` parameter, and set `--preset_lora_model` to the same parameters as `lora_base_model` to load LoRA 1 into the base model.

## Framework Design Concept

In the training framework, the model pointed to by `--preset_lora_path` is loaded in the `switch_pipe_to_training_mode` of `DiffusionTrainingModule`.