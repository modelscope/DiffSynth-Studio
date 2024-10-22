# Text-to-Image, Image-to-Image, and High-Resolution Restoration - First Encounter with the Dazzling Diffusion.

Load the text-to-image model, here we use an anime-style model from Civitai as an example.

```python
import torch
from diffsynth import ModelManager, SDImagePipeline, download_models

download_models(["AingDiffusion_v12"])
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models(["models/stable_diffusion/aingdiffusion_v12.safetensors"])
pipe = SDImagePipeline.from_model_manager(model_manager)
```

Generate a picture to give it a try.

```python
torch.manual_seed(0)
image = pipe(
    prompt="masterpiece, best quality, a girl with long silver hair",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,",
    height=512, width=512, num_inference_steps=80,
)
image.save("image.jpg")
```

Ah, a lovely young lady.

![image](https://github.com/user-attachments/assets/999100d2-1c39-4f18-b37e-aa9d5b4e519c)

Use the image-to-image feature to turn her hair red, simply by adding `input_image` and `denoising_strength` as parameters. The `denoising_strength` controls the intensity of the noise added, when set to 0, the generated image will be identical to the input image, and when set to 1, it will be completely randomly generated.

```python
torch.manual_seed(1)
image_edited = pipe(
    prompt="masterpiece, best quality, a girl with long red hair",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,",
    height=512, width=512, num_inference_steps=80,
    input_image=image, denoising_strength=0.6,
)
image_edited.save("image_edited.jpg")
```

Ah, a cute girl with red hair.

![image_edited](https://github.com/user-attachments/assets/e3de8bc1-037f-4d4d-aacf-8919143c2375)

Since the model itself was trained at a resolution of 512*512, the image appears a bit blurry. However, we can utilize the model's own capabilities to refine the image and add details. Specifically, this involves increasing the resolution and then using image-to-image generation.
```python
torch.manual_seed(2)
image_highres = pipe(
    prompt="masterpiece, best quality, a girl with long red hair",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,",
    height=1024, width=1024, num_inference_steps=80,
    input_image=image_edited.resize((1024, 1024)), denoising_strength=0.6,
)
image_highres.save("image_highres.jpg")
```

Ah, a clear and lovely girl with red hair.

![image_highres](https://github.com/user-attachments/assets/4466353e-662c-49f5-9211-b11bb0bb7fb7)

It's worth noting that the image-to-image and high-resolution restoration features are globally supported, and currently, all of our image generation pipelines can be used in this way.