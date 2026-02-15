import torch
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

prompt = "A solo girl with silver wavy hair and blue eyes, wearing a blue dress, underwater, air bubbles, floating hair."
negative_prompt = "nsfw, low quality"

image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    seed=0, 
    cfg_scale=2, 
    num_inference_steps=50,
    enable_ses=True,
    ses_reward_model="pick",
    ses_eval_budget=20,
    ses_inference_steps=20
)
image.save("flux_ses_optimized.jpg")