from diffsynth.pipelines.sensenova_u1_image import SenseNovaU1ImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = SenseNovaU1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="model*.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="./"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = "A neon bar sign that clearly reads \"OPEN LATE\", dark interior, moody reflections, easy text rendering. Any text in the image must be rendered exactly as written in quotation marks, with correct spelling, clean typography, and strong readability."
image = pipe(prompt=prompt, seed=42, height=1024, width=1024, num_inference_steps=50, cfg_scale=4.0, shift=3.0)
image.save("image_SenseNova-U1.5-8B-MoT.jpg")
