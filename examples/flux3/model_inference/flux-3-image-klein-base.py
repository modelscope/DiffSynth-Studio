from PIL import Image
import torch

from diffsynth.pipelines.flux3_image import Flux3ImagePipeline, ModelConfig

# `black-forest-labs/flux-3-image-klein-base` is an access-request-only model. To download it:
# 1. Request access at https://modelscope.cn/models/black-forest-labs/flux-3-image-klein-base
# 2. Log in via CLI: `modelscope login --token ms-xxx`. Get your token at https://modelscope.cn/my/settings/token
pipe = Flux3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein-base", origin_file_pattern="model.safetensors"),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein-base", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein-base", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="black-forest-labs/flux-3-image-klein-base", origin_file_pattern="text_encoder/"),
)

prompt = "Masterpiece, best quality. Cinematic photograph of a beautiful young woman with long black hair wearing a navy sailor-style school uniform with a green necktie, standing in a sunlit school corridor with large windows, warm afternoon light and soft bokeh in the background."
image_1 = pipe(prompt=prompt, seed=0, num_inference_steps=50, cfg_scale=3.5)
image_1.save("image1.png")

prompt = (
    "Masterpiece, best quality. Cinematic photograph of <girl_1>, a cheerful young woman with a short auburn bob "
    "wearing a white short-sleeved blouse and a deep red pleated skirt with red Mary Jane shoes, standing on a leafy campus path at golden hour, "
    "a brown leather satchel <bag_1> resting on the ground at frame-left and a vintage bicycle <bicycle_1> parked at frame-right, soft bokeh in the background. "
    '[{"id":"girl_1","bbox":[60,330,980,690],"desc":"Cheerful young woman with a short auburn bob, a white short-sleeved blouse and a deep red pleated skirt with red Mary Jane shoes, standing facing the camera."},'
    '{"id":"bag_1","bbox":[740,100,940,360],"desc":"Brown leather school satchel with brass buckles and a shoulder strap, resting on the leaf-covered ground."},'
    '{"id":"bicycle_1","bbox":[380,720,880,980],"desc":"Vintage bicycle with a dark green frame, cream tires and a woven front basket, leaning on its kickstand."}]'
)
image_2 = pipe(prompt=prompt, seed=1, num_inference_steps=50, cfg_scale=3.5)
image_2.save("image2.png")

prompt = "Masterpiece, best quality. Cinematic photograph of the two young women from the two reference images standing side by side at the same scale in a sunlit school courtyard, the long-black-haired woman in the navy sailor school uniform and the short-auburn-haired woman in the white blouse and deep red pleated skirt, warm afternoon light, keeping both women's hairstyles and outfits unchanged."
edit_image = [Image.open("image1.png"), Image.open("image2.png")]
image_3 = pipe(prompt=prompt, edit_image=edit_image, seed=2, num_inference_steps=50, cfg_scale=3.5)
image_3.save("image3.png")
