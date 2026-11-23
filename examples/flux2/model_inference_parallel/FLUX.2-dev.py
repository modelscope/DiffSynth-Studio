# torchrun --standalone --nproc_per_node=8 examples/flux2/model_inference_parallel/FLUX.2-dev.py
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch
from PIL import Image
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="tokenizer/"),
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses")),
)
prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."
image = pipe(prompt, seed=42, rand_device="cuda", num_inference_steps=50)
if dist.get_rank() == 0:
    image.save("image_FLUX.2-dev.jpg")

prompt  = "Transform the image into Japanese anime style"
dist.barrier()
edit_image = [Image.open("image_FLUX.2-dev.jpg")]
image = pipe(prompt, seed=42, rand_device="cuda", edit_image=edit_image, num_inference_steps=50, embedded_guidance=2.5)
if dist.get_rank() == 0:
    image.save("image_FLUX.2-dev_edit.jpg")