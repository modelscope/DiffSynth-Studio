import json
import fnmatch
from pathlib import Path
from typing import Union
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

ImageInput = Union[Image.Image, list[Image.Image], tuple[Image.Image, ...]]

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _image_transform(image_size):
    return Compose(
        [
            Resize(image_size, interpolation=BICUBIC),
            CenterCrop(image_size),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )

def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def _find_file(path, names):
    path = Path(path)
    if path.is_file():
        return path if any(fnmatch.fnmatch(path.name, name) for name in names) else None
    for name in names:
        candidate = path / name
        if candidate.exists():
            return candidate
    for pattern in names:
        matches = sorted(path.rglob(pattern))
        if matches:
            return matches[0]
    return None

class ImageRewardMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
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

        for name, param in self.layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (input_size + 1))
            if "bias" in name:
                nn.init.constant_(param, val=0)

    def forward(self, x):
        return self.layers(x)

class ImageRewardModel(nn.Module):
    def __init__(self, blip, tokenizer, image_size=224, max_length=35, mean=0.16717362830052426, std=1.0333394966054072):
        super().__init__()
        self.blip = blip
        self.tokenizer = tokenizer
        self.preprocess = _image_transform(image_size)
        self.max_length = max_length
        self.mlp = ImageRewardMLP(blip.config.text_config.hidden_size)
        self.register_buffer("score_mean", torch.tensor(float(mean)), persistent=False)
        self.register_buffer("score_std", torch.tensor(float(std)), persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        med_config_path: str = None,
        tokenizer_path: str = None,
        torch_dtype: torch.dtype = None,
        device: Union[str, torch.device] = "cpu",
        max_length: int = 35,
        model_kwargs: dict = None,
        tokenizer_kwargs: dict = None,
    ):
        from transformers import BertTokenizer, BlipConfig, BlipForImageTextRetrieval

        model_kwargs = {} if model_kwargs is None else model_kwargs
        tokenizer_kwargs = {} if tokenizer_kwargs is None else tokenizer_kwargs
        model_path = Path(model_path)
        checkpoint_path = _find_file(model_path, ["ImageReward.pt", "pytorch_model.bin", "*.pt", "*.bin", "*.safetensors"])
        if checkpoint_path is None:
            raise FileNotFoundError(f"Cannot find an ImageReward checkpoint under {model_path}.")

        med_config_path = Path(med_config_path) if med_config_path is not None else _find_file(model_path, ["med_config.json"])
        text_config = cls._load_text_config(med_config_path)
        if tokenizer_path is None:
            if cls._has_tokenizer_files(model_path):
                tokenizer_path = str(model_path)
            else:
                raise ValueError(
                    "ImageReward requires a local BERT tokenizer path. Use "
                    "`ImageRewardMetric.from_pretrained(...)`, or pass a "
                    "ModelScope-downloaded tokenizer such as "
                    "`AI-ModelScope/bert-base-uncased`."
                )
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")

        vision_hidden_size = model_kwargs.pop("vision_hidden_size", 1024)
        config = BlipConfig(
            vision_config={
                "hidden_size": vision_hidden_size,
                "intermediate_size": vision_hidden_size * 4,
                "num_hidden_layers": model_kwargs.pop("vision_num_hidden_layers", 24),
                "num_attention_heads": model_kwargs.pop("vision_num_attention_heads", 16),
                "image_size": model_kwargs.pop("image_size", 224),
                "patch_size": model_kwargs.pop("patch_size", 16),
                "hidden_act": "gelu",
                "layer_norm_eps": model_kwargs.pop("vision_layer_norm_eps", 1e-6),
            },
            text_config={
                **text_config,
                "vocab_size": max(text_config.get("vocab_size", 0), len(tokenizer)),
                "encoder_hidden_size": vision_hidden_size,
                "add_cross_attention": True,
                "is_decoder": True,
            },
            projection_dim=model_kwargs.pop("projection_dim", 256),
        )
        blip = BlipForImageTextRetrieval(config)
        model = cls(blip=blip, tokenizer=tokenizer, max_length=max_length)
        state_dict = cls._load_checkpoint(checkpoint_path)
        converted = cls._convert_state_dict(state_dict)
        model.load_state_dict(converted, strict=False)
        if torch_dtype is not None:
            model.blip = model.blip.to(dtype=torch_dtype)
            model.mlp = model.mlp.float()
        model = model.to(device).eval()
        return model

    @staticmethod
    def _load_text_config(med_config_path):
        if med_config_path is None:
            return {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "max_position_embeddings": 512,
                "vocab_size": 30524,
                "hidden_act": "gelu",
                "layer_norm_eps": 1e-12,
                "attention_probs_dropout_prob": 0.1,
                "hidden_dropout_prob": 0.1,
                "pad_token_id": 0,
                "type_vocab_size": 2,
            }
        with open(med_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        allowed = {
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "max_position_embeddings",
            "vocab_size",
            "hidden_act",
            "layer_norm_eps",
            "attention_probs_dropout_prob",
            "hidden_dropout_prob",
            "pad_token_id",
            "type_vocab_size",
        }
        return {key: value for key, value in data.items() if key in allowed}

    @staticmethod
    def _has_tokenizer_files(path):
        path = Path(path)
        return path.is_dir() and any((path / name).exists() for name in ("vocab.txt", "tokenizer.json", "tokenizer_config.json"))

    @staticmethod
    def _load_checkpoint(checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.suffix == ".safetensors":
            import safetensors.torch

            state_dict = safetensors.torch.load_file(str(checkpoint_path), device="cpu")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
        return state_dict

    @staticmethod
    def _convert_state_dict(state_dict):
        converted = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            new_key, new_value = ImageRewardModel._convert_key_value(key, value)
            if new_key is not None:
                converted[new_key] = new_value
        return converted

    @staticmethod
    def _convert_key_value(key, value):
        if key.startswith("blip.visual_encoder."):
            suffix = key[len("blip.visual_encoder.") :]
            if suffix == "cls_token":
                return "blip.vision_model.embeddings.class_embedding", value
            if suffix == "pos_embed":
                return "blip.vision_model.embeddings.position_embedding", value
            if suffix.startswith("patch_embed.proj."):
                return "blip.vision_model.embeddings.patch_embedding." + suffix[len("patch_embed.proj.") :], value
            if suffix.startswith("blocks."):
                parts = suffix.split(".")
                layer = parts[1]
                rest = ".".join(parts[2:])
                prefix = f"blip.vision_model.encoder.layers.{layer}."
                mapping = {
                    "norm1.": "layer_norm1.",
                    "attn.qkv.": "self_attn.qkv.",
                    "attn.proj.": "self_attn.projection.",
                    "norm2.": "layer_norm2.",
                    "mlp.fc1.": "mlp.fc1.",
                    "mlp.fc2.": "mlp.fc2.",
                }
                for source, target in mapping.items():
                    if rest.startswith(source):
                        return prefix + target + rest[len(source) :], value
            if suffix.startswith("norm."):
                return "blip.vision_model.post_layernorm." + suffix[len("norm.") :], value
            return None, value
        return key, value

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self):
        try:
            return next(self.blip.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _tokenize(self, prompts):
        return self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

    def _preprocess_images(self, images):
        tensors = [self.preprocess(image.convert("RGB")) for image in images]
        return torch.stack(tensors, dim=0).to(device=self.device, dtype=self.dtype)

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
    def forward(self, prompts: Union[str, list[str]], images):
        prompts, images = self._normalize_inputs(prompts, images)
        text_input = self._tokenize(prompts)
        image_tensor = self._preprocess_images(images)
        image_output = self.blip.vision_model(pixel_values=image_tensor, return_dict=True)
        image_embeds = image_output.last_hidden_state
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
        text_output = self.blip.text_encoder(
            input_ids=text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        text_features = text_output.last_hidden_state[:, 0, :].float()
        rewards = self.mlp(text_features).squeeze(-1)
        rewards = (rewards - self.score_mean) / self.score_std
        return rewards
