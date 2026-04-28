from diffsynth.diffusion.template import TemplatePipeline
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
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-PandaMeme")],
    lazy_loading=True,
)
image = template(
    pipe,
    prompt="A meme with a sleepy expression.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{}],
    negative_template_inputs = [{}],
)
image.save("image_PandaMeme_sleepy.jpg")
image = template(
    pipe,
    prompt="A meme with a happy expression.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{}],
    negative_template_inputs = [{}],
)
image.save("image_PandaMeme_happy.jpg")
image = template(
    pipe,
    prompt="A meme with a surprised expression.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{}],
    negative_template_inputs = [{}],
)
image.save("image_PandaMeme_surprised.jpg")
