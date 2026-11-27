# torchrun --standalone --nproc_per_node=8 examples/flux/model_inference_parallel/FLUX.1-Kontext-dev.py
import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from PIL import Image
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-Kontext-dev", origin_file_pattern="flux1-kontext-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses")),
)

image_1 = pipe(
    prompt="a beautiful Asian long-haired female college student.",
    embedded_guidance=2.5,
    seed=1,
)
if dist.get_rank() == 0:
    image_1.save("image_1.jpg")

image_2 = pipe(
    prompt="transform the style to anime style.",
    kontext_images=image_1,
    embedded_guidance=2.5,
    seed=2,
)
if dist.get_rank() == 0:
    image_2.save("image_2.jpg")

image_3 = pipe(
    prompt="let her smile.",
    kontext_images=image_1,
    embedded_guidance=2.5,
    seed=3,
)
if dist.get_rank() == 0:
    image_3.save("image_3.jpg")

image_4 = pipe(
    prompt="let the girl play basketball.",
    kontext_images=image_1,
    embedded_guidance=2.5,
    seed=4,
)
if dist.get_rank() == 0:
    image_4.save("image_4.jpg")

image_5 = pipe(
    prompt="move the girl to a park, let her sit on a chair.",
    kontext_images=image_1,
    embedded_guidance=2.5,
    seed=5,
)
if dist.get_rank() == 0:
    image_5.save("image_5.jpg")