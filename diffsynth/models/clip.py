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


class CLIPModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, processor, max_length: int = 77):
        super().__init__()
        self.model = model
        self.processor = processor
        self.max_length = max_length

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        processor_path: str = None,
        torch_dtype: torch.dtype = None,
        device: Union[str, torch.device] = "cuda",
        max_length: int = 77,
        model_kwargs: dict = None,
        processor_kwargs: dict = None,
    ):
        from modelscope import AutoModel, AutoProcessor

        model_kwargs = {} if model_kwargs is None else model_kwargs
        processor_kwargs = {} if processor_kwargs is None else processor_kwargs
        processor_path = model_path if processor_path is None else processor_path
        processor = AutoProcessor.from_pretrained(processor_path, **processor_kwargs)
        model = AutoModel.from_pretrained(model_path, **model_kwargs).eval()
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        model = model.to(device)
        return cls(model=model, processor=processor, max_length=max_length)

    @property
    def device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self):
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

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
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return image_features

    @torch.no_grad()
    def get_text_features(self, text: Union[str, list[str]]):
        text_inputs = self._processor_call(text=text)
        text_features = _feature_tensor(self.model.get_text_features(**text_inputs), "text_features")
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return text_features

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
