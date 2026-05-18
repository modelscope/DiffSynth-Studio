from pathlib import Path
from typing import Union
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

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


def _find_checkpoint(path, version):
    path = Path(path)
    version_to_file = {
        "v2.0": "HPS_v2_compressed.pt",
        "v2.1": "HPS_v2.1_compressed.pt",
    }
    if path.is_file():
        return path
    filename = version_to_file.get(version)
    names = [filename] if filename is not None else []
    names += ["*.pt", "*.pth", "*.bin", "*.safetensors"]
    for name in names:
        if name is None:
            continue
        candidate = path / name
        if candidate.exists():
            return candidate
        matches = sorted(path.rglob(name))
        if matches:
            return matches[0]
    return None

class HPSv2Model(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        processor_path: str,
        version: str = "v2.0",
        torch_dtype: torch.dtype = None,
        device: Union[str, torch.device] = "cpu",
        model_kwargs: dict = None,
        processor_kwargs: dict = None,
    ):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        processor_kwargs = {} if processor_kwargs is None else processor_kwargs
        processor = AutoProcessor.from_pretrained(processor_path, **processor_kwargs)
        checkpoint_path = _find_checkpoint(model_path, version)
        config = AutoConfig.from_pretrained(processor_path)
        model = AutoModel.from_config(config, **model_kwargs)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Cannot find an HPSv2 checkpoint under {model_path}.")
        state_dict = cls._load_checkpoint(checkpoint_path)
        state_dict = cls._prepare_state_dict(state_dict, model.state_dict())
        model.load_state_dict(state_dict, strict=False)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        model = model.to(device).eval()
        return cls(model=model, processor=processor)

    @staticmethod
    def _load_checkpoint(checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
        return {key[len("module.") :] if key.startswith("module.") else key: value for key, value in state_dict.items()}

    @staticmethod
    def _prepare_state_dict(state_dict, target_state_dict):
        converted = {}
        for key, value in state_dict.items():
            updates = HPSv2Model._convert_open_clip_key(key, value)
            for new_key, new_value in updates:
                if new_key in target_state_dict and tuple(target_state_dict[new_key].shape) == tuple(new_value.shape):
                    converted[new_key] = new_value
        return converted

    @staticmethod
    def _convert_open_clip_key(key, value):
        if key == "logit_scale":
            return [("logit_scale", value)]
        if key == "token_embedding.weight":
            return [("text_model.embeddings.token_embedding.weight", value)]
        if key == "positional_embedding":
            return [("text_model.embeddings.position_embedding.weight", value)]
        if key.startswith("ln_final."):
            return [("text_model.final_layer_norm." + key[len("ln_final.") :], value)]
        if key == "text_projection":
            return [("text_projection.weight", value.T)]
        if key == "visual.class_embedding":
            return [("vision_model.embeddings.class_embedding", value)]
        if key == "visual.conv1.weight":
            return [("vision_model.embeddings.patch_embedding.weight", value)]
        if key == "visual.positional_embedding":
            return [("vision_model.embeddings.position_embedding.weight", value)]
        if key.startswith("visual.ln_pre."):
            return [("vision_model.pre_layrnorm." + key[len("visual.ln_pre.") :], value)]
        if key.startswith("visual.ln_post."):
            return [("vision_model.post_layernorm." + key[len("visual.ln_post.") :], value)]
        if key == "visual.proj":
            return [("visual_projection.weight", value.T)]
        if key.startswith("transformer.resblocks."):
            return HPSv2Model._convert_resblock("text_model.encoder.layers", key[len("transformer.resblocks.") :], value)
        if key.startswith("visual.transformer.resblocks."):
            return HPSv2Model._convert_resblock("vision_model.encoder.layers", key[len("visual.transformer.resblocks.") :], value)
        return []

    @staticmethod
    def _convert_resblock(prefix, suffix, value):
        parts = suffix.split(".")
        layer = parts[0]
        rest = ".".join(parts[1:])
        layer_prefix = f"{prefix}.{layer}."
        if rest == "attn.in_proj_weight":
            q, k, v = value.chunk(3, dim=0)
            return [
                (layer_prefix + "self_attn.q_proj.weight", q),
                (layer_prefix + "self_attn.k_proj.weight", k),
                (layer_prefix + "self_attn.v_proj.weight", v),
            ]
        if rest == "attn.in_proj_bias":
            q, k, v = value.chunk(3, dim=0)
            return [
                (layer_prefix + "self_attn.q_proj.bias", q),
                (layer_prefix + "self_attn.k_proj.bias", k),
                (layer_prefix + "self_attn.v_proj.bias", v),
            ]
        mapping = {
            "attn.out_proj.": "self_attn.out_proj.",
            "ln_1.": "layer_norm1.",
            "ln_2.": "layer_norm2.",
            "mlp.c_fc.": "mlp.fc1.",
            "mlp.c_proj.": "mlp.fc2.",
        }
        for source, target in mapping.items():
            if rest.startswith(source):
                return [(layer_prefix + target + rest[len(source) :], value)]
        return []

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
        inputs = self.processor(text=prompts, images=images, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)
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
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scores = (image_features * text_features).sum(dim=-1)
        if hasattr(self.model, "logit_scale"):
            scores = self.model.logit_scale.exp() * scores
        return scores
