# torchrun --standalone --nproc_per_node=8 examples/qwen_image/model_inference_parallel/Qwen-Image-Edit-2509.py
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses")),
)

image_1 = pipe(prompt="一位少女", seed=0, num_inference_steps=40, height=1328, width=1024)
if dist.get_rank() == 0:
    image_1.save("image1.jpg")

image_2 = pipe(prompt="一位老人", seed=0, num_inference_steps=40, height=1328, width=1024)
if dist.get_rank() == 0:
    image_2.save("image2.jpg")

prompt = "生成这两个人的合影"
dist.barrier()
edit_image = [Image.open("image1.jpg"), Image.open("image2.jpg")]
image_3 = pipe(prompt, edit_image=edit_image, seed=1, num_inference_steps=40, height=1328, width=1024, edit_image_auto_resize=True)
if dist.get_rank() == 0:
    image_3.save("image3.jpg")

# Qwen-Image-Edit-2509 is a multi-image editing model.
# Please use a list to input `edit_image`, even if the input contains only one image.
# edit_image = [Image.open("image.jpg")]
# Please do not input the image directly.
# edit_image = Image.open("image.jpg")
