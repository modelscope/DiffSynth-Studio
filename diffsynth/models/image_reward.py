from typing import Union
import torch
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


class ImageRewardMLP(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
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

        for name, param in self.layers.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0.0, std=1.0 / (input_size + 1))
            if "bias" in name:
                torch.nn.init.constant_(param, val=0)

    def forward(self, x):
        return self.layers(x)


class ImageRewardModel(torch.nn.Module):
    def __init__(self, blip=None, tokenizer=None, image_size=224, max_length=35, mean=0.16717362830052426, std=1.0333394966054072):
        super().__init__()
        if blip is None:
            blip = self.default_blip_model()
            
        self.blip = blip
        self.tokenizer = tokenizer
        self.preprocess = _image_transform(image_size)
        self.max_length = max_length
        self.mlp = ImageRewardMLP(blip.config.text_config.hidden_size)
        
        self.register_buffer("score_mean", torch.tensor(float(mean)), persistent=False)
        self.register_buffer("score_std", torch.tensor(float(std)), persistent=False)

    @staticmethod
    def default_blip_model():
        from transformers import BlipConfig, BlipForImageTextRetrieval

        vision_hidden_size = 1024
        text_config = ImageRewardModel._load_text_config(None)
        config = BlipConfig(
            vision_config={
                "hidden_size": vision_hidden_size,
                "intermediate_size": vision_hidden_size * 4,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "image_size": 224,
                "patch_size": 16,
                "hidden_act": "gelu",
                "layer_norm_eps": 1e-6,
            },
            text_config={
                **text_config,
                "vocab_size": 30524,
                "encoder_hidden_size": vision_hidden_size,
                "add_cross_attention": True,
                "is_decoder": True,
            },
            projection_dim=256,
        )
        return BlipForImageTextRetrieval(config)

    @staticmethod
    def _load_text_config(med_config_path):
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

    @staticmethod
    def convert_key_value(key, value):
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
        return next(self.parameters(), torch.tensor([])).device

    @property
    def dtype(self):
        return next(self.parameters(), torch.tensor(0.0)).dtype

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
        
        text_features = text_output.last_hidden_state[:, 0, :]
        rewards = self.mlp(text_features).squeeze(-1)
        rewards = (rewards - self.score_mean) / self.score_std
        
        return rewards