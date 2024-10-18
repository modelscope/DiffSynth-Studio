#Training FLUX LoRA

The following files will be used to build the FLUX model. You can download them from [huggingface](https://huggingface.co/black-forest-labs/FLUX.1-dev)或[modelscope](https://www.modelscope.cn/models/ai-modelscope/flux.1-dev), or you can use the following code to download these files:
```python
from diffsynth import download_models

download_models(["FLUX.1-dev"])
```

```
models/FLUX/
└── FLUX.1-dev
    ├── ae.safetensors
    ├── flux1-dev.safetensors
    ├── text_encoder
    │   └── model.safetensors
    └── text_encoder_2
        ├── config.json
        ├── model-00001-of-00002.safetensors
        ├── model-00002-of-00002.safetensors
        └── model.safetensors.index.json
```

Start the training task with the following command:

```
CUDA_VISIBLE_DEVICES="0" python examples/train/flux/train_flux_lora.py \
  --pretrained_text_encoder_path models/FLUX/FLUX.1-dev/text_encoder/model.safetensors \
  --pretrained_text_encoder_2_path models/FLUX/FLUX.1-dev/text_encoder_2 \
  --pretrained_dit_path models/FLUX/FLUX.1-dev/flux1-dev.safetensors \
  --pretrained_vae_path models/FLUX/FLUX.1-dev/ae.safetensors \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 500 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "bf16" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing
```

For more information on the parameters, please use `python examples/train/flux/train_flux_lora.py -h` to view detailed information.

After the training is complete, use `model_manager.load_lora` to load the LoRA for inference.

```python
from diffsynth import ModelManager, FluxImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=[
                                 "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
                                 "models/FLUX/FLUX.1-dev/text_encoder_2",
                                 "models/FLUX/FLUX.1-dev/ae.safetensors",
                                 "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
                             ])
model_manager.load_lora("models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = SDXLImagePipeline.from_model_manager(model_manager)

torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5
)
image.save("image_with_lora.jpg")
```
