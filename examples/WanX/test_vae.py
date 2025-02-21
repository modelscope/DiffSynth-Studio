import torch
import torchvision
import imageio
from diffsynth import ModelManager

def save_video(tensor,
                save_file=None,
                fps=30,
                nrow=8,
                normalize=True,
                value_range=(-1, 1)):

    tensor = tensor.clamp(min(value_range), max(value_range))
    tensor = torch.stack([
        torchvision.utils.make_grid(
            u, nrow=nrow, normalize=normalize, value_range=value_range)
        for u in tensor.unbind(2)
    ],
                         dim=1).permute(1, 2, 3, 0) #frame, h, w, 3
    tensor = (tensor * 255).type(torch.uint8).cpu()

    # write video
    writer = imageio.get_writer(
        save_file, fps=fps, codec='libx264', quality=8)
    for frame in tensor.numpy():
        writer.append_data(frame)
    writer.close()

torch.cuda.memory._record_memory_history()

model_manager = ModelManager(torch_dtype=torch.float, device="cuda")
model_manager.load_models([
    "models/WanX/vae.pth",
])

vae = model_manager.fetch_model('wanxvideo_vae')

latents = [torch.load('sample.pt')]
videos = vae.decode(latents, device=latents[0].device, tiled=True)
# back_encode = vae.encode(videos)

torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

save_video(videos[0][None], save_file='example3.mp4', fps=16, nrow=8)
print(latents)
print(videos)
# print(back_encode)
