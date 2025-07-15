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
        ModelConfig(model_id="DiffSynth-Studio/FLUX.1-dev-ValueController", origin_file_pattern="single/prefer_embed/value.ckpt", offload_device="cpu", offload_dtype=torch.float8_e4m3fn)
    ],
)
pipe.load_lora(pipe.dit, ModelConfig(model_id="DiffSynth-Studio/FLUX.1-dev-ValueController", origin_file_pattern="single/dit_lora/dit_value.ckpt"))
pipe.enable_vram_management()

for i in range(10):
    image = pipe(prompt="a cat", seed=0, value_controller_inputs=[i/10])
    image.save(f"value_control_{i}.jpg")