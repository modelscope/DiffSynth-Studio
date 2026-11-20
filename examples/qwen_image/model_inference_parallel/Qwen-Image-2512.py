# torchrun --standalone --nproc_per_node=8 examples/qwen_image/model_inference_parallel/Qwen-Image-2512.py
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-2512", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses")),
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40)
if dist.get_rank() == 0:
    image.save("image.jpg")
