from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from PIL import Image
import torch


pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Omni-Base", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Omni-Base", origin_file_pattern="siglip/model.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)
prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
image = pipe(prompt=prompt, seed=0, num_inference_steps=40, cfg_scale=4)
image.save("image_Z-Image-Omni-Base.jpg")

image = Image.open("image_Z-Image-Omni-Base.jpg")
prompt = "Change the women's clothes to white cheongsam, keep other content unchanged"
image = pipe(prompt=prompt, edit_image=image, seed=42, rand_device="cuda", num_inference_steps=40, cfg_scale=4)
image.save("image_edit_Z-Image-Omni-Base.jpg")
