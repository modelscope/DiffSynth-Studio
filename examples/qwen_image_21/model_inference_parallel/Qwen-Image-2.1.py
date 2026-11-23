# torchrun --standalone --nproc_per_node=8 examples/qwen_image_21/model_inference_parallel/Qwen-Image-2.1.py
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
import torch
from PIL import Image
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig

pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses")),
)

# Text-to-Image, the output is an RGBA image
prompt = "Flat anime-style illustration, a girl with long black hair, wearing a JK uniform."
image = pipe(prompt, seed=0)
if dist.get_rank() == 0:
    image.save("image1.png")

prompt = "Flat anime-style illustration, a sunny and cheerful high school girl."
image_2 = pipe(prompt=prompt, seed=0)
if dist.get_rank() == 0:
    image_2.save("image2.png")

# Image Editing, the generated RGBA image is fed back as the condition
prompt = "Generate a group photo of these two characters."
dist.barrier()
edit_image = [Image.open("image1.png"), Image.open("image2.png")]
image_3 = pipe(prompt, edit_image=edit_image, seed=1)
if dist.get_rank() == 0:
    image_3.save("image3.png")
