import torch
from PIL import Image
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="model*.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="edit_decoder.bin"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/FLUX.1-NexusGen-Edit_lora/epoch-4.safetensors", alpha=1)

ref_image = Image.open("data/example_image_dataset/nexus_gen/image_1.png").convert("RGB")
prompt = "Add a pair of sunglasses."
image = pipe(
    prompt=prompt, negative_prompt="",
    seed=42, cfg_scale=1.0, num_inference_steps=50,
    nexus_gen_reference_image=ref_image,
    height=512, width=512,
)
image.save("NexusGen-Edit_lora.jpg")
