from typing import Union
import torch
from PIL import Image

ImageInput = Union[Image.Image, list[Image.Image], tuple[Image.Image, ...]]

class AestheticMLP(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def _as_image_list(images: ImageInput):
    if isinstance(images, Image.Image):
        images = [images]
    return [image.convert("RGB") for image in images]


class AestheticModel(torch.nn.Module):
    def __init__(
        self, 
        mlp: AestheticMLP = None,
        vision_model: torch.nn.Module = None, 
        visual_projection: torch.nn.Module = None, 
        processor=None
    ):
        super().__init__()
        if vision_model is None:
            vision_model, visual_projection = self.default_vision_model()
        if mlp is None:
            mlp = AestheticMLP(768)
            
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.processor = processor
        self.layers = mlp.layers

    @staticmethod
    def default_vision_model():
        from transformers import CLIPVisionConfig, CLIPVisionModel

        config = CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_hidden_layers=24,
            image_size=224,
            patch_size=14,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            projection_dim=768,
        )
        return CLIPVisionModel(config), torch.nn.Linear(config.hidden_size, config.projection_dim, bias=False)

    @property
    def device(self):
        return next(self.parameters(), torch.tensor([])).device

    @property
    def dtype(self):
        return next(self.parameters(), torch.tensor(0.0)).dtype

    @torch.no_grad()
    def get_image_features(self, images):            
        images = _as_image_list(images)
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        
        image_features = self.vision_model(pixel_values=pixel_values, return_dict=True).pooler_output
        image_features = self.visual_projection(image_features)
        
        return torch.nn.functional.normalize(image_features, dim=-1)

    @torch.no_grad()
    def forward(self, images):
        image_features = self.get_image_features(images)
        return self.layers(image_features).squeeze(-1)