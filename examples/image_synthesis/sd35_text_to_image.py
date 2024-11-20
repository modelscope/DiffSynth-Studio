from diffsynth import ModelManager, SD3ImagePipeline
import torch


model_manager = ModelManager(torch_dtype=torch.float16, device="cuda", model_id_list=["StableDiffusion3.5-large"])
pipe = SD3ImagePipeline.from_model_manager(model_manager)

prompt = "a full body photo of a beautiful Asian girl. CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

torch.manual_seed(1)
image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    cfg_scale=5,
    num_inference_steps=100, width=1024, height=1024,
)
image.save("image_1024.jpg")

image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    cfg_scale=5,
    input_image=image.resize((2048, 2048)), denoising_strength=0.5,
    num_inference_steps=50, width=2048, height=2048,
    tiled=True
)
image.save("image_2048.jpg")
