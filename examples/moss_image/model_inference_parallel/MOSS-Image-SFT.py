# torchrun --standalone --nproc_per_node=8 examples/moss_image/model_inference_parallel/MOSS-Image-SFT.py
from diffsynth.pipelines.moss_image import MOSSImagePipeline, ModelConfig
import torch
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig

pipe = MOSSImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", computation_dtype=torch.float32),
    ],
    tokenizer_config=ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="tokenizer/"),
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses")),
)

prompt = '''A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:
"记得买牛奶和鸡蛋！
牛奶两盒，鸡蛋一打。
顺便带一袋面包。"'''

image = pipe(prompt, height=1024, width=1024, seed=42, num_inference_steps=50, cfg_scale=3.0)
if dist.get_rank() == 0:
    image.save("image_MOSS-Image-SFT.jpg")
