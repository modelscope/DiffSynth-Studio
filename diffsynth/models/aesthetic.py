from pathlib import Path
from typing import Union
import json
import torch
import torch.nn as nn
from PIL import Image
from .clip import CLIPModel

ImageInput = Union[Image.Image, list[Image.Image], tuple[Image.Image, ...]]

class AestheticMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def _as_image_list(images: Union[ImageInput, list[ImageInput], tuple[ImageInput, ...]]):
    if isinstance(images, Image.Image):
        images = [images]
    return [image.convert("RGB") for image in images]


class AestheticModel(torch.nn.Module):
    def __init__(self, mlp: AestheticMLP, clip_model: CLIPModel = None, vision_model: torch.nn.Module = None, processor=None):
        super().__init__()
        self.clip_model = clip_model
        self.vision_model = vision_model
        self.processor = processor
        self.mlp = mlp

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        clip_model_path: str = None,
        clip_processor_path: str = None,
        torch_dtype: torch.dtype = None,
        device: Union[str, torch.device] = "cpu",
        clip_kwargs: dict = None,
        processor_kwargs: dict = None,
    ):
        checkpoint = cls._load_checkpoint(model_path)
        model = cls._from_full_predictor(
            model_path=model_path,
            checkpoint=checkpoint,
            torch_dtype=torch_dtype,
            device=device,
            processor_kwargs=processor_kwargs,
        )
        return model

    @classmethod
    def _from_full_predictor(
        cls,
        model_path: str,
        checkpoint: dict,
        torch_dtype: torch.dtype = None,
        device: Union[str, torch.device] = "cuda",
        processor_kwargs: dict = None,
    ):
        from transformers import AutoProcessor, CLIPVisionModelWithProjection

        processor_kwargs = {} if processor_kwargs is None else processor_kwargs
        config = cls._load_vision_config(model_path)
        vision_model = CLIPVisionModelWithProjection(config)
        mlp = AestheticMLP(config.projection_dim)
        normalized = cls._normalize_checkpoint_keys(checkpoint)
        vision_state = {}
        mlp_state = {}
        for key, value in normalized.items():
            if key.startswith("layers."):
                mlp_state[key] = value
            elif key in vision_model.state_dict():
                vision_state[key] = value
        if not vision_state:
            raise ValueError(f"Cannot find CLIP vision tower weights in Aesthetic checkpoint under {model_path}.")
        vision_model.load_state_dict(vision_state, strict=False)
        mlp.load_state_dict(mlp_state, strict=True)
        processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        if torch_dtype is not None:
            vision_model = vision_model.to(dtype=torch_dtype)
        vision_model = vision_model.to(device).eval()
        mlp = mlp.to(device).float().eval()
        return cls(vision_model=vision_model, processor=processor, mlp=mlp).eval()

    @staticmethod
    def _load_vision_config(model_path):
        from transformers import CLIPVisionConfig

        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot find Aesthetic config.json under {model_path}.")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config_data = data.get("vision_config", data)
        if "projection_dim" not in config_data and "projection_dim" in data:
            config_data = dict(config_data)
            config_data["projection_dim"] = data["projection_dim"]
        allowed = {
            "attention_dropout",
            "dropout",
            "hidden_act",
            "hidden_size",
            "image_size",
            "initializer_factor",
            "initializer_range",
            "intermediate_size",
            "layer_norm_eps",
            "num_attention_heads",
            "num_channels",
            "num_hidden_layers",
            "patch_size",
            "projection_dim",
        }
        return CLIPVisionConfig(**{key: value for key, value in config_data.items() if key in allowed})

    @staticmethod
    def _find_checkpoint(path):
        path = Path(path)
        if path.is_file():
            return path
        names = [
            "model.safetensors",
            "pytorch_model.bin",
            "sac+logos+ava1-l14-linearMSE.pth",
            "ava+logos-l14-linearMSE.pth",
            "*.pth",
            "*.pt",
            "*.bin",
            "*.safetensors",
        ]
        for name in names:
            candidate = path / name
            if candidate.exists():
                return candidate
            matches = sorted(path.rglob(name))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"Cannot find an Aesthetic MLP checkpoint under {path}.")

    @classmethod
    def _load_checkpoint(cls, path):
        checkpoint_path = cls._find_checkpoint(path)
        if checkpoint_path.suffix == ".safetensors":
            import safetensors.torch

            checkpoint = safetensors.torch.load_file(str(checkpoint_path), device="cpu")
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        return checkpoint

    @staticmethod
    def _normalize_checkpoint_keys(checkpoint):
        normalized = {}
        for key, value in checkpoint.items():
            for prefix in ("model.", "module.", "aesthetic_model.", "aesthetics_predictor.", "predictor."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            normalized[key] = value
        return normalized
        
    @property
    def device(self):
        if self.clip_model is not None:
            return self.clip_model.device
        try:
            return next(self.vision_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self):
        if self.clip_model is not None:
            return self.clip_model.dtype
        try:
            return next(self.vision_model.parameters()).dtype
        except StopIteration:
            return torch.float32

    @torch.no_grad()
    def get_image_features(self, images):
        if self.clip_model is not None:
            return self.clip_model.get_image_features(images)
        images = _as_image_list(images)
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        image_features = self.vision_model(pixel_values=pixel_values, return_dict=True).image_embeds
        return image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def forward(self, images):
        image_features = self.get_image_features(images).float()
        return self.mlp(image_features).squeeze(-1)
