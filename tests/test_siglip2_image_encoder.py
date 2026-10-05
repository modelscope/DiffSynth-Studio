import importlib

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import SiglipVisionConfig, SiglipVisionModel

from diffsynth.configs.model_configs import MODEL_CONFIGS
from diffsynth.core.loader.model import load_model
from diffsynth.models.siglip2_image_encoder import Siglip2ImageEncoder


@pytest.fixture
def tiny_config(monkeypatch):
    config = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, image_size=384, patch_size=16,
        _attn_implementation="sdpa",
    )
    monkeypatch.setattr(
        "transformers.models.siglip.modeling_siglip.SiglipVisionConfig",
        lambda **kwargs: config,
    )
    return config


def test_siglip_forward_matches_public_model(tiny_config):
    model = Siglip2ImageEncoder().eval()
    image = Image.new("RGB", (64, 48), color=(32, 64, 128))
    pixels = model.processor(images=[image], return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        expected = SiglipVisionModel.forward(model, pixel_values=pixels).pooler_output
        actual = model(image, torch_dtype=torch.float32, device="cpu")
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, tiny_config.hidden_size)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("use_disk_map", [False, True])
def test_siglip_loads_bare_vision_checkpoint(tiny_config, tmp_path, use_disk_map):
    reference = SiglipVisionModel(tiny_config).eval()
    weights = reference.vision_model.state_dict()
    descriptor = next(c for c in MODEL_CONFIGS if c["model_name"] == "siglip2_image_encoder")
    converter_path = descriptor.get("state_dict_converter")
    converter = None
    if converter_path:
        module_name, name = converter_path.rsplit(".", 1)
        converter = getattr(importlib.import_module(module_name), name)
    path = tmp_path / "model.safetensors"
    save_file(weights, path)
    model = load_model(
        Siglip2ImageEncoder, path=str(path), use_disk_map=use_disk_map,
        torch_dtype=torch.float32, device="cpu", state_dict_converter=converter,
    )
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value)
