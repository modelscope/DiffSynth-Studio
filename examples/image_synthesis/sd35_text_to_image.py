from diffsynth import ModelManager, SD3ImagePipeline
import torch



model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["StableDiffusion3.5-large"])
pipe = SD3ImagePipeline.from_model_manager(model_manager)

prompt = "A capybara holding a sign that reads Hello World"
negative_prompt = ""

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=3.5,
    num_inference_steps=28, width=1024, height=1024,
    seed=0
)
image.save("image_1024.jpg")
