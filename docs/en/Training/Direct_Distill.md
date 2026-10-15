# End-to-End Direct Distillation Acceleration (Deprecated)

(At this stage, we strongly recommend that developers use OPD for distillation accelerated training. The content in this document is for research purposes only.)

## Algorithm Principle

In the training framework of `DiffSynth-Studio`, we designed an end-to-end distillation accelerated training scheme, which we call Direct Distillation. The pseudocode for the training process is as follows:

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

First, you need to generate training data. Please refer to the [Model Inference](../Pipeline_Usage/Model_Inference.md) section to write inference code and generate training data with a sufficient number of inference steps.

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

Then, we compile the necessary information into [metadata files](../API_Reference/core/data.md#metadata):

```csv
image,prompt,seed,rand_device,num_inference_steps,cfg_scale
distill_qwen/image.jpg,"精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。",0,cpu,4,1
```

This sample dataset can be downloaded directly:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

Then start LoRA distillation accelerated training:

```shell
bash examples/qwen_image/model_training/lora/Qwen-Image-Distill-LoRA.sh
```

Please note that in the [training script parameters](../Pipeline_Usage/Model_Training.md#script-parameters), the image resolution setting for the dataset should avoid triggering scaling processing. When setting `--height` and `--width` to enable fixed resolution, all training data must be generated with exactly the same width and height. When setting `--max_pixels` to enable dynamic resolution, the value of `--max_pixels` must be greater than or equal to the pixel area of any training image.
