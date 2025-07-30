import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import load_state_dict


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/LoRA-Encoder-FLUX.1-Dev", origin_file_pattern="model.safetensors"),
    ],
)
pipe.enable_lora_magic()
state_dict = load_state_dict("models/train/FLUX.1-dev-LoRA-Encoder_full/epoch-0.safetensors")
pipe.lora_encoder.load_state_dict(state_dict)

lora = ModelConfig(model_id="VoidOc/flux_animal_forest1", origin_file_pattern="20.safetensors")
pipe.load_lora(pipe.dit, lora, hotload=True) # Use `pipe.clear_lora()` to drop the loaded LoRA.

image = pipe(prompt="", seed=0, lora_encoder_inputs=lora)
image.save("image_FLUX.1-dev-LoRA-Encoder_full.jpg")
