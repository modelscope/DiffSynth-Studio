import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth import load_state_dict


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/AttriCtrl-FLUX.1-Dev", origin_file_pattern="models/brightness.safetensors")
    ],
)
state_dict = load_state_dict("models/train/FLUX.1-dev-AttriCtrl_full/epoch-0.safetensors")
pipe.value_controller.encoders[0].load_state_dict(state_dict)

image = pipe(prompt="a cat", seed=0, value_controller_inputs=0.1, rand_device="cuda")
image.save("image_FLUX.1-dev-AttriCtrl_full.jpg")
