# torchrun --standalone --nproc_per_node=8 examples/ernie_image/model_inference_parallel/ERNIE-Image.py
from diffsynth.pipelines.ernie_image import ErnieImagePipeline, ModelConfig
import torch
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig, CFGParallelConfig

pipe = ErnieImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device='cuda',
    model_configs=[
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="tokenizer/"),
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses"), cfg=CFGParallelConfig()),
)

image = pipe(
    prompt="一只黑白相间的中华田园犬",
    negative_prompt="",
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=50,
    cfg_scale=4.0,
)
if dist.get_rank() == 0:
    image.save("output.jpg")
