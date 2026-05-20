from typing import Union
import torch
from PIL import Image

ImageInput = Union[Image.Image, list[Image.Image], tuple[Image.Image, ...]]

def _feature_tensor(output, feature_name: str):
    if torch.is_tensor(output):
        return output
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    if isinstance(output, (list, tuple)):
        for value in output:
            if torch.is_tensor(value):
                return value
    raise TypeError(f"{feature_name} must be a tensor or a model output with projected features.")


class PickScoreModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, processor, max_length: int = 77):
        super().__init__()
        self.model = model
        self.processor = processor
        self.max_length = max_length

    @property
    def device(self):
        return next(self.parameters(), torch.tensor([])).device

    @property
    def dtype(self):
        return next(self.parameters(), torch.tensor(0.0)).dtype

    def _processor_call(self, **kwargs):
        inputs = self.processor(
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.device)
        
        if self.dtype != torch.float32:
            inputs = {
                name: (
                    value.to(dtype=self.dtype)
                    if torch.is_tensor(value) and torch.is_floating_point(value)
                    else value
                )
                for name, value in inputs.items()
            }
            
        return inputs

    @torch.no_grad()
    def get_image_features(self, images: ImageInput):
        if isinstance(images, Image.Image):
            images = [images]
        images = [image.convert("RGB") for image in images]
        
        image_inputs = self._processor_call(images=images)
        image_features = _feature_tensor(self.model.get_image_features(**image_inputs), "image_features")
        
        return torch.nn.functional.normalize(image_features, dim=-1)

    @torch.no_grad()
    def get_text_features(self, text: Union[str, list[str]]):
        text_inputs = self._processor_call(text=text)
        text_features = _feature_tensor(self.model.get_text_features(**text_inputs), "text_features")
        
        return torch.nn.functional.normalize(text_features, dim=-1)

    @torch.no_grad()
    def forward(self, text: Union[str, list[str]], images: ImageInput):
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(text)
        
        scores = self.model.logit_scale.exp() * (text_features @ image_features.T)
        if isinstance(text, str):
            scores = scores[0]
            
        return scores