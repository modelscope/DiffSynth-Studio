import torch
from PIL import Image
from modelscope import dataset_snapshot_download
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, ControlNetInput


vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint", origin_file_pattern="model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="inpaint/*.jpg"
)
prompt = "a cat with sunglasses"
controlnet_image = Image.open("./data/example_image_dataset/inpaint/image_1.jpg").convert("RGB").resize((1328, 1328))
inpaint_mask = Image.open("./data/example_image_dataset/inpaint/mask.jpg").convert("RGB").resize((1328, 1328))
image = pipe(
    prompt, seed=0,
    input_image=controlnet_image, inpaint_mask=inpaint_mask,
    blockwise_controlnet_inputs=[ControlNetInput(image=controlnet_image, inpaint_mask=inpaint_mask)],
    num_inference_steps=40,
)
image.save("image.jpg")
