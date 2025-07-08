import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig, ControlNetInput
from diffsynth import load_state_dict
from PIL import Image


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="ByteDance/InfiniteYou", origin_file_pattern="infu_flux_v1.0/aes_stage2/image_proj_model.bin"),
        ModelConfig(model_id="ByteDance/InfiniteYou", origin_file_pattern="infu_flux_v1.0/aes_stage2/InfuseNetModel/*.safetensors"),
    ],
)
state_dict = load_state_dict("models/train/FLUX.1-dev-InfiniteYou_full/epoch-0.safetensors")
state_dict_projector = {i.replace("image_proj_model.", ""): state_dict[i] for i in state_dict if i.startswith("image_proj_model.")}
pipe.image_proj_model.load_state_dict(state_dict_projector)
state_dict_controlnet = {i.replace("controlnet.models.0.", ""): state_dict[i] for i in state_dict if i.startswith("controlnet.models.0.")}
pipe.controlnet.models[0].load_state_dict(state_dict_controlnet)

image = pipe(
    prompt="a man with a red hat",
    controlnet_inputs=[ControlNetInput(
        image=Image.open("data/example_image_dataset/infiniteyou/image_1.jpg"),
    )],
    height=1024, width=1024,
    seed=0, rand_device="cuda",
)
image.save("image_FLUX.1-dev-InfiniteYou_full.jpg")
