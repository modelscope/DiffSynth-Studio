from diffsynth.pipelines.qwen_image import (
    QwenImagePipeline, ModelConfig,
    QwenImageUnit_Image2LoRAEncode, QwenImageUnit_Image2LoRADecode
)
from diffsynth.utils.lora import merge_lora
from diffsynth import load_state_dict
from modelscope import snapshot_download
from safetensors.torch import save_file
import torch
from PIL import Image


def demo_style():
    # Load models
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="SigLIP2-G384/model.safetensors"),
            ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="DINOv3-7B/model.safetensors"),
            ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-i2L", origin_file_pattern="Qwen-Image-i2L-Style.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )

    # Load images
    snapshot_download(
        model_id="DiffSynth-Studio/Qwen-Image-i2L",
        allow_file_pattern="assets/style/1/*",
        local_dir="data/examples"
    )
    images = [
        Image.open("data/examples/assets/style/1/0.jpg"),
        Image.open("data/examples/assets/style/1/1.jpg"),
        Image.open("data/examples/assets/style/1/2.jpg"),
        Image.open("data/examples/assets/style/1/3.jpg"),
        Image.open("data/examples/assets/style/1/4.jpg"),
    ]

    # Model inference
    with torch.no_grad():
        embs = QwenImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=images)
        lora = QwenImageUnit_Image2LoRADecode().process(pipe, **embs)["lora"]
    save_file(lora, "model_style.safetensors")


def demo_coarse_fine_bias():
    # Load models
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="SigLIP2-G384/model.safetensors"),
            ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="DINOv3-7B/model.safetensors"),
            ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-i2L", origin_file_pattern="Qwen-Image-i2L-Coarse.safetensors"),
            ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-i2L", origin_file_pattern="Qwen-Image-i2L-Fine.safetensors"),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )

    # Load images
    snapshot_download(
        model_id="DiffSynth-Studio/Qwen-Image-i2L",
        allow_file_pattern="assets/lora/3/*",
        local_dir="data/examples"
    )
    images = [
        Image.open("data/examples/assets/lora/3/0.jpg"),
        Image.open("data/examples/assets/lora/3/1.jpg"),
        Image.open("data/examples/assets/lora/3/2.jpg"),
        Image.open("data/examples/assets/lora/3/3.jpg"),
        Image.open("data/examples/assets/lora/3/4.jpg"),
        Image.open("data/examples/assets/lora/3/5.jpg"),
    ]

    # Model inference
    with torch.no_grad():
        embs = QwenImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=images)
        lora = QwenImageUnit_Image2LoRADecode().process(pipe, **embs)["lora"]
        lora_bias = ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-i2L", origin_file_pattern="Qwen-Image-i2L-Bias.safetensors")
        lora_bias.download_if_necessary()
        lora_bias = load_state_dict(lora_bias.path, torch_dtype=torch.bfloat16, device="cuda")
        lora = merge_lora([lora, lora_bias])
    save_file(lora, "model_coarse_fine_bias.safetensors")


def generate_image(lora_path, prompt, seed):
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
    pipe.load_lora(pipe.dit, lora_path)
    image = pipe(prompt, seed=seed, height=1024, width=1024, num_inference_steps=50)
    return image


demo_style()
image = generate_image("model_style.safetensors", "a cat", 0)
image.save("image_1.jpg")

demo_coarse_fine_bias()
image = generate_image("model_coarse_fine_bias.safetensors", "bowl", 1)
image.save("image_2.jpg")
