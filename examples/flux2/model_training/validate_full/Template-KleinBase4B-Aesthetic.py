from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
import torch

pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
)
pipe.dit = pipe.enable_lora_hot_loading(pipe.dit) # Important!
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-Aesthetic")],
)
state_dict = load_state_dict("./models/train/Template-KleinBase4B-Aesthetic_full/epoch-1.safetensors", torch_dtype=torch.bfloat16)
template.models[0].load_state_dict(state_dict)
image = template(
    pipe,
    prompt="a bird with fire",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "lora_ids": [1],
        "lora_scales": 1.0,
        "merge_type": "mean",
    }],
    negative_template_inputs = [{
        "lora_ids": [1],
        "lora_scales": 1.0,
        "merge_type": "mean",
    }],
)
image.save("image_Aesthetic_1.0.jpg")
image = template(
    pipe,
    prompt="a bird with fire",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "lora_ids": [1],
        "lora_scales": 2.5,
        "merge_type": "mean",
    }],
    negative_template_inputs = [{
        "lora_ids": [1],
        "lora_scales": 2.5,
        "merge_type": "mean",
    }],
)
image.save("image_Aesthetic_2.5.jpg")
