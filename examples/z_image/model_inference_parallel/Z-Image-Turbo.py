# torchrun --standalone --nproc_per_node=6 examples/z_image/model_inference_parallel/Z-Image-Turbo.py
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
import torch
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig


pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses")),
)
prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
image = pipe(prompt=prompt, seed=42, rand_device="cuda")
if dist.get_rank() == 0:
    image.save("image_Z-Image-Turbo.jpg")
