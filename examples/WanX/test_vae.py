import torch
from diffsynth.models.wanx_vae import WanXVideoVAE
from diffsynth import ModelManager


model_manager = ModelManager(torch_dtype=torch.float, device="cuda")
model_manager.load_models([
    "models/WanX/vae.pth",
])

vae = model_manager.fetch_model('wanxvideo_vae')

latents = [torch.load('sample.pt')]
videos = vae.decode(latents)
back_encode = vae.encode(videos)
print(latents)
print(videos)
print(back_encode)
