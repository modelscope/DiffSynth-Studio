import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)

pipe.load_timestep_lora(ModelConfig(model_id="DiffSynth-Studio/MultiAlign-FLUX.1-dev", origin_file_pattern="adapter_*"))

prompt = "A moonlit Venetian canal scene with a gondola floating on the left, a candlelit table for two on a stone terrace to the right, and flower boxes under arched windows in the peach-colored building behind, with shimmering reflections in dark water; romantic cinematic mood and the restaurant sign reading 'NOTTE SERENA'"

image = pipe(prompt=prompt, seed=0, num_inference_steps=30, t5_sequence_length=128)
image.save("FLUX.1-dev-Timestep-LoRA.jpg")
