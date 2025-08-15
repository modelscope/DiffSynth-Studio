from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import snapshot_download
import torch

# Please do not use float8 on this model
snapshot_download("DiffSynth-Studio/Qwen-Image-Distill-LoRA", local_dir="models/DiffSynth-Studio/Qwen-Image-Distill-LoRA")
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", offload_device="cpu"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.enable_vram_management()
pipe.load_lora(pipe.dit, "models/DiffSynth-Studio/Qwen-Image-Distill-LoRA/model.safetensors")

prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=15, cfg_scale=1)
image.save("image.jpg")