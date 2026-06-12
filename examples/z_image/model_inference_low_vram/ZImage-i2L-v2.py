from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from modelscope import snapshot_download
from PIL import Image
import numpy as np
import torch

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
pipe.enable_lora_hot_loading(pipe.dit)
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    lazy_loading=True,
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/ZImage-i2L-v2")],
)
snapshot_download("DiffSynth-Studio/ZImage-i2L-v2", allow_file_pattern="assets/*", local_dir="data")
images = [Image.open(f"data/assets/multi_input_{i}.jpg") for i in range(4)]
image = template(
    pipe,
    prompt="A cat is sitting on a stone",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{"image": images}],
    negative_template_inputs = [{"image": [Image.fromarray(np.zeros_like(np.array(i)) + 128) for i in images]}],
)
image.save("image_output.jpg")