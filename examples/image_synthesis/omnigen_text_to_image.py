import torch
from diffsynth import ModelManager, OmnigenImagePipeline


model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["OmniGen-v1"])
pipe = OmnigenImagePipeline.from_model_manager(model_manager)

image_man = pipe(
    prompt="A portrait of a man.",
    cfg_scale=2.5, num_inference_steps=50, seed=0
)
image_man.save("image_man.jpg")

image_woman = pipe(
    prompt="A portrait of an Asian woman with a white t-shirt.",
    cfg_scale=2.5, num_inference_steps=50, seed=1
)
image_woman.save("image_woman.jpg")

image_merged = pipe(
    prompt="a man and a woman. The man is the man in <img><|image_1|></img>. The woman is the woman in <img><|image_2|></img>.",
    reference_images=[image_man, image_woman],
    cfg_scale=2.5, image_cfg_scale=2.5, num_inference_steps=50, seed=2
)
image_merged.save("image_merged.jpg")
