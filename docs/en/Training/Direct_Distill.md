# End-to-End Distillation Accelerated Training

## Distillation Accelerated Training

The inference process of Diffusion models typically requires multi-step iterations, which improves generation quality but also makes the generation process slow. Through distillation accelerated training, the number of steps required to generate clear content can be reduced. The essence of distillation accelerated training technology is to align the generation effects of a small number of steps with those of a large number of steps.

There are diverse methods for distillation accelerated training, such as:

* Adversarial training ADD (Adversarial Diffusion Distillation)
    * Paper: https://arxiv.org/abs/2311.17042
    * Model: [stabilityai/sdxl-turbo](https://modelscope.cn/models/stabilityai/sdxl-turbo)
* Progressive training Hyper-SD
    * Paper: https://arxiv.org/abs/2404.13686
    * Model: [ByteDance/Hyper-SD](https://www.modelscope.cn/models/ByteDance/Hyper-SD)

## Direct Distillation

At the framework level, supporting these distillation accelerated training schemes is extremely difficult. In the design of the training framework, we need to ensure that the training scheme meets the following conditions:

* Generality: The training scheme applies to most Diffusion models supported within the framework, rather than only working for a specific model, which is a basic requirement for code framework construction.
* Stability: The training scheme must ensure stable training effects without requiring manual fine-tuning of parameters. Adversarial training in ADD cannot guarantee stability.
* Simplicity: The training scheme does not introduce additional complex modules. According to Occam's Razor principle, complex solutions may introduce potential risks. The Human Feedback Learning in Hyper-SD makes the training process overly complex.

Therefore, in the training framework of `DiffSynth-Studio`, we designed an end-to-end distillation accelerated training scheme, which we call Direct Distillation. The pseudocode for the training process is as follows:

```
seed = xxx
with torch.no_grad():
    image_1 = pipe(prompt, steps=50, seed=seed, cfg=4)
image_2 = pipe(prompt, steps=4, seed=seed, cfg=1)
loss = torch.nn.functional.mse_loss(image_1, image_2)
```

Yes, it's a very end-to-end training scheme that produces immediate results with minimal training.

## Models Trained with Direct Distillation

We trained two models based on Qwen-Image using this scheme:

* [DiffSynth-Studio/Qwen-Image-Distill-Full](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full): Full distillation training
* [DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA): LoRA distillation training

Click on the model links to go to the model pages and view the model effects.

## Using Distillation Accelerated Training in the Training Framework

First, you need to generate training data. Please refer to the [Model Inference](/docs/en/Pipeline_Usage/Model_Inference.md) section to write inference code and generate training data with a sufficient number of inference steps.

Taking Qwen-Image as an example, the following code can generate an image:

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

Then, we compile the necessary information into [metadata files](/docs/en/API_Reference/core/data.md#metadata):

```csv
image,prompt,seed,rand_device,num_inference_steps,cfg_scale
distill_qwen/image.jpg,"精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。",0,cpu,4,1
```

This sample dataset can be downloaded directly:

```shell
modelscope download --dataset DiffSynth-Studio/example_image_dataset --local_dir ./data/example_image_dataset
```

Then start LoRA distillation accelerated training:

```shell
bash examples/qwen_image/model_training/lora/Qwen-Image-Distill-LoRA.sh
```

Please note that in the [training script parameters](/docs/en/Pipeline_Usage/Model_Training.md#script-parameters), the image resolution setting for the dataset should avoid triggering scaling processing. When setting `--height` and `--width` to enable fixed resolution, all training data must be generated with exactly the same width and height. When setting `--max_pixels` to enable dynamic resolution, the value of `--max_pixels` must be greater than or equal to the pixel area of any training image.

## Framework Design Concept

Compared to [Standard Supervised Training](/docs/en/Training/Supervised_Fine_Tuning.md), Direct Distillation only differs in the training loss function. The loss function for Direct Distillation is `DirectDistillLoss` in `diffsynth.diffusion.loss`.

## Future Work

Direct Distillation is a highly general acceleration scheme, but it may not be the best-performing scheme. Therefore, we have not yet published this technology in paper form. We hope to leave this problem to the academic and open-source communities to solve together, and we look forward to developers providing more complete general training schemes.