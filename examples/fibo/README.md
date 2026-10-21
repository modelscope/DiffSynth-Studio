English Document: https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Fibo.html

中文文档：https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Fibo.html

## Inference

- Text-to-image, editing, masked editing, text rendering: [`model_inference/Fibo-2.0-dev.py`](model_inference/Fibo-2.0-dev.py)
- Editing and masked editing: [`model_inference/Fibo-2.0-edit-turbo.py`](model_inference/Fibo-2.0-edit-turbo.py)
- Low VRAM versions: [`model_inference_low_vram/`](model_inference_low_vram/)

## Training

Only Fibo-2.0-dev is trainable.

| Mode | Full Training | LoRA Training | Full Validation | LoRA Validation |
|-|-|-|-|-|
| Text-to-image | [`full/Fibo-2.0-dev-T2I.sh`](model_training/full/Fibo-2.0-dev-T2I.sh) | [`lora/Fibo-2.0-dev-T2I.sh`](model_training/lora/Fibo-2.0-dev-T2I.sh) | [`validate_full/Fibo-2.0-dev-T2I.py`](model_training/validate_full/Fibo-2.0-dev-T2I.py) | [`validate_lora/Fibo-2.0-dev-T2I.py`](model_training/validate_lora/Fibo-2.0-dev-T2I.py) |
| Image editing | [`full/Fibo-2.0-dev-Edit.sh`](model_training/full/Fibo-2.0-dev-Edit.sh) | [`lora/Fibo-2.0-dev-Edit.sh`](model_training/lora/Fibo-2.0-dev-Edit.sh) | [`validate_full/Fibo-2.0-dev-Edit.py`](model_training/validate_full/Fibo-2.0-dev-Edit.py) | [`validate_lora/Fibo-2.0-dev-Edit.py`](model_training/validate_lora/Fibo-2.0-dev-Edit.py) |
