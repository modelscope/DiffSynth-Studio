import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-Krea-dev", origin_file_pattern="flux1-krea-dev.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
    ],
)
pipe.enable_vram_management()

prompt = "An beautiful woman is riding a bicycle in a park, wearing a red dress"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

image = pipe(prompt=prompt, seed=0, embedded_guidance=4.5)
image.save("flux_krea.jpg")

image = pipe(
    prompt=prompt, negative_prompt=negative_prompt,
    seed=0, cfg_scale=2, num_inference_steps=50,
    embedded_guidance=4.5
)
image.save("flux_krea_cfg.jpg")
