from typing import Union
import torch
from PIL import Image
from transformers import CLIPModel as HFCLIPModel

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


class ImageMetricsCLIPModel(HFCLIPModel):
    def __init__(self, variant: str = "h14"):
        super().__init__(self.config(variant))

    @staticmethod
    def config(variant: str):
        from transformers import CLIPConfig
        return CLIPConfig(
            projection_dim=1024,
            logit_scale_init_value=2.6592,
            text_config={
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "max_position_embeddings": 77,
                "vocab_size": 49408,
                "hidden_act": "quick_gelu",
                "layer_norm_eps": 1e-5,
                "projection_dim": 1024,
                "bos_token_id": 0,
                "eos_token_id": 2,
                "pad_token_id": 1,
            },
            vision_config={
                "hidden_size": 1280,
                "intermediate_size": 5120,
                "num_attention_heads": 16,
                "num_hidden_layers": 32,
                "image_size": 224,
                "patch_size": 14,
                "hidden_act": "quick_gelu",
                "layer_norm_eps": 1e-5,
                "projection_dim": 1024,
            },
        )
        raise ValueError(f"Unsupported ImageMetrics CLIP variant: {variant}")


class CLIPModel(torch.nn.Module):
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

    def _normalize_pairs(self, text, images):
        if isinstance(text, str):
            text = [text]
        else:
            text = list(text)
           
        if isinstance(images, Image.Image):
            images = [images]
        images = [image.convert("RGB") for image in images]
        
        if len(text) == 1 and len(images) > 1:
            text = text * len(images)
        if len(images) == 1 and len(text) > 1:
            images = images * len(text)
            
        if len(text) != len(images):
            raise ValueError(f"Expected the same number of prompts and images, got {len(text)} and {len(images)}.")
        return text, images

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
    def similarity_matrix(self, text: Union[str, list[str]], images: ImageInput):
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(text)
        
        scores = text_features @ image_features.T
        if hasattr(self.model, "logit_scale"):
            scores = self.model.logit_scale.exp() * scores
        return scores

    @torch.no_grad()
    def forward(self, text: Union[str, list[str]], images: ImageInput):
        text, images = self._normalize_pairs(text, images)
        
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(text)
        
        scores = (text_features * image_features).sum(dim=-1)
        if hasattr(self.model, "logit_scale"):
            scores = self.model.logit_scale.exp() * scores
        return scores