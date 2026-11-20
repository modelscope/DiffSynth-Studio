# torchrun --standalone --nproc_per_node=7 examples/boogu_image/model_inference_parallel/Boogu-Image-0.1-Base.py
from diffsynth.pipelines.boogu_image import BooguImagePipeline, ModelConfig
import torch
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig


pipe = BooguImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="mllm/*.safetensors"),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="vae/*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="mllm/"),
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses")),
)

output = pipe(
    prompt="a cat",
    negative_prompt="",
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=50,
    cfg_scale=4.0,
)
if dist.get_rank() == 0:
    output.save("image_Boogu-Image-0.1-Base.jpg")
