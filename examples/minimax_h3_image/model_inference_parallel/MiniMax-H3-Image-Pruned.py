# torchrun --standalone --nproc_per_node=8 examples/minimax_h3_image/model_inference_parallel/MiniMax-H3-Image-Pruned.py
import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from PIL import Image
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig


vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack", origin_file_pattern="models/bf16_table_pruned.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="pdd_fused_heads.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses")),
)

prompt = (
    "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough "
    "on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts, "
    "warm afternoon light raking across the grain of the wood."
)
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
if dist.get_rank() == 0:
    image.save("MiniMax-H3-Image-Pruned.jpg")

prompt = "Change the color of the tomato in <Picture 1> to green."
dist.barrier()
image = pipe(
    prompt=prompt, height=4096, width=4096, seed=43,
    edit_image=[Image.open("MiniMax-H3-Image-Pruned.jpg")],
)
if dist.get_rank() == 0:
    image.save("MiniMax-H3-Image-Pruned-Edit.jpg")
