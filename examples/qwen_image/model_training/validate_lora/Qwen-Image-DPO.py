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
pipe.load_lora(pipe.dit, "models/train/Qwen-Image_DPO_lora/epoch-4.safetensors")
prompt = "黑板上写着“群起效尤,心灵手巧”，字的颜色分别是 “群”: 橙色、“起”: 黑色、“效”: 蓝色、“尤”: 绿色、“心”: 紫色、“灵”: 粉色、“手”: 红色、“巧”: 白色"
for seed in range(0, 5):
    image = pipe(prompt, seed=seed)
    image.save(f"image_dpo_{seed}.jpg")
