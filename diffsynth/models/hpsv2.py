from typing import Union
import torch
from PIL import Image

ImageInput = Union[Image.Image, list[Image.Image], tuple[Image.Image, ...]]

def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

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


class HPSv2Model(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    @property
    def device(self):
        return next(self.parameters(), torch.tensor([])).device

    @property
    def dtype(self):
        return next(self.parameters(), torch.tensor(0.0)).dtype

    def _normalize_inputs(self, prompts, images):
        images = _as_list(images)
        prompts = _as_list(prompts)
        
        if len(prompts) == 1 and len(images) > 1:
            prompts = prompts * len(images)
        if len(images) == 1 and len(prompts) > 1:
            images = images * len(prompts)
            
        if len(prompts) != len(images):
            raise ValueError(f"Expected the same number of prompts and images, got {len(prompts)} and {len(images)}.")
        return prompts, images

    @torch.no_grad()
    def forward(self, prompts: Union[str, list[str]], images: ImageInput):
        prompts, images = self._normalize_inputs(prompts, images)
        images = [image.convert("RGB") for image in images]
        
        inputs = self.processor(
            text=prompts, 
            images=images, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
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
            
        image_features = _feature_tensor(
            self.model.get_image_features(pixel_values=inputs["pixel_values"]),
            "image_features",
        )
        text_features = _feature_tensor(
            self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")),
            "text_features",
        )
        
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        
        scores = (image_features * text_features).sum(dim=-1)
        if hasattr(self.model, "logit_scale"):
            scores = self.model.logit_scale.exp() * scores
            
        return scores