# Distillation Accelerated Training

## What Is Distillation Accelerated Training?

The inference process of Diffusion models typically requires multi-step iterations, which improves generation quality but also makes the generation process slow. Through distillation accelerated training, the number of steps required to generate clear content can be reduced. The essence of distillation accelerated training technology is to align the generation effects of a small number of steps with those of a large number of steps.

In the design of the training framework, we need to evaluate the feasibility of a distillation accelerated training scheme from the following aspects:

* Generality: The training scheme applies to most Diffusion models supported within the framework, rather than only working for a specific model, which is a basic requirement for code framework construction.
* Stability: The training scheme must ensure stable training effects without requiring manual fine-tuning of parameters.
* Simplicity: The training scheme does not introduce additional complex modules. According to Occam's Razor principle, complex solutions may introduce potential risks.
* Generation Quality: The model obtained from training produces generation results at a small number of steps similar to those of the original model at a large number of steps.

We have tried several distillation accelerated training schemes, and the empirical results are shown in the following table:

|Training Scheme|Generality|Stability|Simplicity|Generation Quality|
|-|-|-|-|-|
|Direct Distill|★★★|★★★|★★★|★|
|DMD2|★|★|★|★★★|
|OPD (Recommended)|★★★|★★|★★★|★★★|

> **At this stage, we strongly recommend that developers use OPD for distillation accelerated training.**

## Direct Distill

Direct Distill is an end-to-end direct distillation training scheme that we designed. It uses the original model to generate sampling trajectories, and then trains the model to approximate the sampling trajectory with a small number of steps. For details, see [End-to-End Direct Distillation Acceleration](./Direct_Distill.md).

## DMD2

[DMD2](https://arxiv.org/abs/2405.14867) (Distribution Matching Distillation) is a product that combines the theories of Diffusion and GAN. Its training relies on a manually constructed discriminator model. We have only adapted DMD2 training on the FLUX.2 model. For details, see the [example code](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/flux2/model_training/special/dmd2).

## OPD

OPD (On-Policy Distillation) is an online distillation training scheme, including [DiffusionOPD](https://arxiv.org/abs/2605.15055), [DanceOPD](https://arxiv.org/abs/2606.27377), etc. Its algorithm pseudocode is as follows:

```python
with torch.no_grad():
    trajectory = sample_trajectory(student_model)
input = trajectory[randint]
student_output = student_model(input)
with torch.no_grad():
    teacher_output = teacher_model(output) # Multi-step inference
loss = torch.nn.functional.mse(student_output, teacher_output)
```

In the DiffusionOPD variant, `input = trajectory[randint]` is replaced with a for loop, using every point in the trajectory for training. In the DanceOPD variant, `input` is sampled only in the low-noise-content stage, avoiding a large divergence in the model's output after training. In our example code, these two variants can be implemented respectively via the parameters `--opd_num_distill_steps` and `--opd_min_distill_idx`.

Compared with Direct Distill, OPD samples and updates starting from the student model rather than pre-generating trajectories with the teacher model, which makes the trained model perform better. Compared with DMD2, OPD does not require designing a discriminator model based on the model structure, so it is highly general and does not require careful tuning of various hyperparameters. Using FLUX.2, Qwen-Image-2.1, and MOSS-Image as examples, we provide example code for OPD training:

|Base Model|Trainable Parameters|Training Code|Post-Training Validation|Steps Before Distillation|Steps After Distillation|
|-|-|-|-|-|-|
|[black-forest-labs/FLUX.2-klein-base-4B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/opd/train_FLUX.2-klein-base-4B-8steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/opd/valudate_FLUX.2-klein-base-4B-8steps_lora.py)|50|8|
|[black-forest-labs/FLUX.2-klein-base-4B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B)|Full|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/opd/train_FLUX.2-klein-base-4B-8steps_full.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/opd/valudate_FLUX.2-klein-base-4B-8steps_full.py)|50|8|
|[Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/train_Qwen-Image-2.1-8steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/validate_Qwen-Image-2.1-8steps_lora.py)|50|8|
|[Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1)|Full|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/train_Qwen-Image-2.1-8steps_full.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/validate_Qwen-Image-2.1-8steps_full.py)|50|8|
|[Qwen/Qwen-Image-2.1-Turbo](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/train_Qwen-Image-2.1-Turbo-3steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/validate_Qwen-Image-2.1-Turbo-3steps_lora.py)|8|3|
|[openmoss/MOSS-Image-SFT](https://www.modelscope.cn/models/openmoss/MOSS-Image-SFT)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/opd/train_MOSS-Image-SFT-8steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/opd/validate_MOSS-Image-SFT-8steps_lora.py)|50|8|
|[openmoss/MOSS-Image-SFT](https://www.modelscope.cn/models/openmoss/MOSS-Image-SFT)|Full|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/opd/train_MOSS-Image-SFT-8steps_full.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/opd/validate_MOSS-Image-SFT-8steps_full.py)|50|8|

If needed, you can easily replicate the example code to implement OPD training for other models. Thanks to the general design of the framework, you can even train with datasets in arbitrary input formats (for example, mixing text-to-image and image-editing datasets), and you can even design new Adapter architecture models specifically for step distillation.

[Qwen-Image-2.1-TurTurbo](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-2.1-TurTurbo) is a distillation accelerated model trained by us, used to verify the effectiveness of the training schemes in the framework. The following practical experiences we obtained from our experiments are for reference:

* OPD converges very quickly during training, and usually produces results within 2 GPU*h.
* In LoRA training, the LoRA should cover the input layer, the output layer, and the layers that process the timestep, because distillation accelerated training greatly changes the output form of the model.
* For LoRA models obtained from OPD training, in a few cases, appropriately increasing the LoRA weight (multiplying the weights by a coefficient greater than 1) can better stimulate the model's capabilities.
* The OPD training process is not absolutely stable; errors generated during sampling may accumulate inside the model. Therefore, longer training is not always better. When necessary, you can train multiple times and average the weights to increase stability.
* When the difference in the number of steps between the teacher model and the student model is very large (for example, 50 and 4), the training results are not good. In this case, you can consider progressive distillation (for example, first distill from 50 to 8, then use the 8-step model as the teacher model to distill to 4).
* The sigmas distribution of the student model is related to the generated image style. When sigmas stay longer in the high-noise stage, the image content tends to become simpler; when sigmas stay longer in the low-noise stage, the image details tend to become richer.
* Enabling CFG can reduce the diversity of generated content, thereby reducing the difficulty of training.

## Future Work

Distillation accelerated training still has great room for improvement in stability and model quality. We will continue to optimize the training effectiveness of the algorithms in the framework.
