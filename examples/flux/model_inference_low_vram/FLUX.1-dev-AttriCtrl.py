import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="DiffSynth-Studio/AttriCtrl-FLUX.1-Dev", origin_file_pattern="models/brightness.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn)
    ],
)
pipe.enable_vram_management()

for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
    image = pipe(prompt="a cat on the beach", seed=2, value_controller_inputs=[i])
    image.save(f"value_control_{i}.jpg")
