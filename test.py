import torch


data = torch.load("models/train/Qwen-Image_lora_cache/0/0.pth", map_location="cpu", weights_only=False)
for i in data:
    print(i)