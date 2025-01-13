import torch
from diffsynth import ModelManager, FluxImagePipeline


model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev"])
pipe = FluxImagePipeline.from_model_manager(model_manager)

prompt = "CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."

for tea_cache_l1_thresh in [None, 0.2, 0.4, 0.6, 0.8]:
    image = pipe(
        prompt=prompt, embedded_guidance=3.5, seed=0,
        num_inference_steps=50, tea_cache_l1_thresh=tea_cache_l1_thresh
    )
    image.save(f"image_{tea_cache_l1_thresh}.png")
