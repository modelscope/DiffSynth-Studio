from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch
from diffsynth.core.quant import QuantizeConfig

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
    "quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw"),
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
    vram_limit=0,
)
pipe.dit = pipe.enable_lora_hot_loading(pipe.dit) # Important!

template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/TreeAdapter-KleinBase4B", origin_file_pattern="iNaturalist/")],
    lazy_loading=True,
)
name = "Glareola pratincola"
prompt = "A small bird with a long tail and short wings stands on sandy ground. Its plumage is light brown above, white below, with a dark collar around its neck. The background is a blurred expanse of sand."
image = template(
    pipe,
    seed=0, cfg_scale=4, num_inference_steps=40,
    template_inputs = [{"name": name, "prompt": prompt}],
    negative_template_inputs = [{"name": name}],
)
image.save("image.jpg")
