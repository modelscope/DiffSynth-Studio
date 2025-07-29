import torch
from PIL import Image
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import load_state_dict

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="model*.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="edit_decoder.bin"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)
state_dict = load_state_dict("models/train/FLUX.1-NexusGen-Edit_full/epoch-0.safetensors")
pipe.dit.load_state_dict(state_dict)

ref_image = Image.open("data/example_image_dataset/nexus_gen/image_1.png").convert("RGB")
prompt = "Add a pair of sunglasses."
image = pipe(
    prompt=prompt, negative_prompt="",
    seed=42, cfg_scale=2.0, num_inference_steps=50,
    nexus_gen_reference_image=ref_image,
    height=512, width=512,
)
image.save("NexusGen-Edit_full.jpg")
