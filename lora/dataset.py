import torch, os
import pandas as pd
from PIL import Image
from torchvision.transforms import v2
from diffsynth.data.video import crop_and_resize


class LoraDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch=1000, loras_per_item=1):
        self.base_path = base_path
        data_df = pd.read_csv(metadata_path)
        self.model_file = data_df["model_file"].tolist()
        self.image_file = data_df["image_file"].tolist()
        self.text = data_df["text"].tolist()
        self.max_resolution = 1920 * 1080
        self.steps_per_epoch = steps_per_epoch
        self.loras_per_item = loras_per_item
        
        
    def read_image(self, image_file):
        image = Image.open(image_file).convert("RGB")
        width, height = image.size
        if width * height > self.max_resolution:
            scale = (width * height / self.max_resolution) ** 0.5
            image = image.resize((int(width / scale), int(height / scale)))
            width, height = image.size
        if width % 16 != 0 or height % 16 != 0:
            image = crop_and_resize(image, height // 16 * 16, width // 16 * 16)
        image = v2.functional.to_image(image)
        image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)
        image = v2.functional.normalize(image, [0.5], [0.5])
        return image
    
    
    def get_data(self, data_id):
        data = {
            "model_file": os.path.join(self.base_path, self.model_file[data_id]),
            "image": self.read_image(os.path.join(self.base_path, self.image_file[data_id])),
            "text": self.text[data_id]
        }
        return data


    def __getitem__(self, index):
        data = []
        while len(data) < self.loras_per_item:
            data_id = torch.randint(0, len(self.model_file), (1,))[0]
            data_id = (data_id + index) % len(self.model_file) # For fixed seed.
            data.append(self.get_data(data_id))
        return data


    def __len__(self):
        return self.steps_per_epoch