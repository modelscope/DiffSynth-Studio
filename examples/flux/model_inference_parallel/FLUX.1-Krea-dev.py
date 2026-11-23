# torchrun --standalone --nproc_per_node=8 examples/flux/model_inference_parallel/FLUX.1-Krea-dev.py
import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-Krea-dev", origin_file_pattern="flux1-krea-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses")),
)

prompt = "An beautiful woman is riding a bicycle in a park, wearing a red dress"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

image = pipe(prompt=prompt, seed=0, embedded_guidance=4.5)
if dist.get_rank() == 0:
    image.save("flux_krea.jpg")

image = pipe(
    prompt=prompt, negative_prompt=negative_prompt,
    seed=0, cfg_scale=2, num_inference_steps=50,
    embedded_guidance=4.5
)
if dist.get_rank() == 0:
    image.save("flux_krea_cfg.jpg")
