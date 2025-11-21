import importlib
import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig


if importlib.util.find_spec("transformers") is None:
    raise ImportError("You are using Nexus-GenV2. It depends on transformers, which is not installed. Please install it with `pip install transformers==4.49.0`.")
else:
    import transformers
    assert transformers.__version__ == "4.49.0", "Nexus-GenV2 requires transformers==4.49.0, please install it with `pip install transformers==4.49.0`."


vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="model*.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="generation_decoder.bin", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vram_config),
    ],
    nexus_gen_processor_config=ModelConfig("DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="processor"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = "一只可爱的猫咪"
image = pipe(
    prompt=prompt, negative_prompt="",
    seed=0, cfg_scale=3, num_inference_steps=50,
    height=1024, width=1024,
)
image.save("cat.jpg")
