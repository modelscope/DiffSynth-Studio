import torch
from PIL import Image
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)

pipe.load_lora(pipe.dit, "models/train/FLUX.1-dev-EliGen_lora/epoch-4.safetensors", alpha=1)

entity_prompts = ["A beautiful girl", "sign 'Entity Control'", "shorts", "shirt"]
global_prompt = "A beautiful girl wearing shirt and shorts in the street,  holding a sign 'Entity Control'"
masks = [Image.open(f"data/example_image_dataset/eligen/{i}.png").convert('RGB') for i in range(len(entity_prompts))]
# generate image
image = pipe(
    prompt=global_prompt,
    cfg_scale=1.0,
    num_inference_steps=50,
    embedded_guidance=3.5,
    seed=42,
    height=1024,
    width=1024,
    eligen_entity_prompts=entity_prompts,
    eligen_entity_masks=masks,
)
image.save(f"EliGen_lora.png")
