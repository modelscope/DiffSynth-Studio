from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, ControlNetInput
from PIL import Image
import torch
from modelscope import dataset_snapshot_download


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny", origin_file_pattern="model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="canny/image_1.jpg"
)
controlnet_image = Image.open("data/example_image_dataset/canny/image_1.jpg").resize((1328, 1328))

prompt = "一只小狗，毛发光洁柔顺，眼神灵动，背景是樱花纷飞的春日庭院，唯美温馨。"
image = pipe(
    prompt, seed=0,
    blockwise_controlnet_inputs=[ControlNetInput(image=controlnet_image)]
)
image.save("image.jpg")
