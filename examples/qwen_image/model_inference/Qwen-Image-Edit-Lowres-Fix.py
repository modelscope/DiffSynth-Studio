from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
from modelscope import snapshot_download

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
snapshot_download("DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix", local_dir="models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix", allow_file_pattern="model.safetensors")
pipe.load_lora(pipe.dit, "models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix/model.safetensors")

prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt=prompt, seed=0, num_inference_steps=40, height=1024, width=768)
image.save("image.jpg")

prompt = "将裙子变成粉色"
image = image.resize((512, 384))
image = pipe(prompt, edit_image=image, seed=1, num_inference_steps=40, height=1024, width=768, edit_rope_interpolation=True, edit_image_auto_resize=False)
image.save(f"image2.jpg")
