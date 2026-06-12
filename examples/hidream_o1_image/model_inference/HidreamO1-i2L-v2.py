from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline, ModelConfig
from modelscope import snapshot_download
from PIL import Image
import numpy as np
import torch

pipe = HiDreamO1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="HiDream-ai/HiDream-O1-Image", origin_file_pattern="model-*.safetensors")],
    processor_config=ModelConfig(model_id="HiDream-ai/HiDream-O1-Image", origin_file_pattern="./"),
)
pipe.enable_lora_hot_loading(pipe.dit)
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/HidreamO1-i2L-v2")],
)
snapshot_download("DiffSynth-Studio/HidreamO1-i2L-v2", allow_file_pattern="assets/*", local_dir="data")
images = [Image.open(f"data/assets/multi_input_{i}.jpg") for i in range(4)]
image = template(
    pipe,
    prompt="A cat is sitting on a stone",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{"image": images}],
    negative_template_inputs = [{"image": [Image.fromarray(np.zeros_like(np.array(i)) + 128) for i in images]}],
)
image.save("image_output.jpg")