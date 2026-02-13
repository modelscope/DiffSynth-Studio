from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch


vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
prompt = "Masterpiece, best quality. Anime-style portrait of a woman in a blue dress, underwater, surrounded by colorful bubbles."
image = pipe(prompt, seed=0, rand_device="cuda", num_inference_steps=4)
image.save("image_FLUX.2-klein-4B.jpg")

prompt = "change the color of the clothes to red"
image = pipe(prompt, edit_image=[image], seed=1, rand_device="cuda", num_inference_steps=4)
image.save("image_edit_FLUX.2-klein-4B.jpg")
