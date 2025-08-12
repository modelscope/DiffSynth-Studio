from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig, ControlNetInput
from diffsynth import load_state_dict
import torch
from PIL import Image
from diffsynth.controlnets.processors import Annotator
import os


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ModelConfig(path="models/DiffSynth-Studio/BlockWiseControlnet/model_init.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)

state_dict = load_state_dict("models/train/Qwen-Image-BlockWiseControlNet_full_lr1e-3_wd1e-6/step-26000.safetensors")
pipe.blockwise_controlnet.load_state_dict(state_dict)

prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = Image.open("test_image.jpg").convert("RGB").resize((1024, 1024))
canny_image = Annotator("canny")(image)
canny_image.save("canny_image_test.jpg")

controlnet_input = ControlNetInput(
    image=canny_image,
    scale=1.0,
    processor_id="canny",
)

for seed in range(100, 200):
    image = pipe(prompt, seed=seed, height=1024, width=1024, controlnet_inputs=[controlnet_input], num_inference_steps=30, cfg_scale=4.0)
    image.save(f"test_image_controlnet_step2k_1_{seed}.jpg")
