import unittest
from unittest.mock import patch

import torch

from diffsynth.pipelines import flux2_image


class _ModelPool:
    def __init__(self, text_encoder=None, text_encoder_qwen3=None):
        self.models = {
            "flux2_text_encoder": text_encoder,
            "z_image_text_encoder": text_encoder_qwen3,
            "flux2_dit": None,
            "flux2_vae": None,
        }

    def fetch_model(self, name):
        return self.models[name]


class _TokenizerConfig:
    path = "tokenizer-path"

    def __init__(self):
        self.downloaded = False

    def download_if_necessary(self):
        self.downloaded = True


class Flux2TokenizerSelectionTest(unittest.TestCase):
    def _load_pipeline(self, model_pool, tokenizer_config):
        with (
            patch.object(
                flux2_image.Flux2ImagePipeline,
                "download_and_load_models",
                return_value=model_pool,
            ),
            patch.object(
                flux2_image.Flux2ImagePipeline,
                "check_vram_management_state",
                return_value=False,
            ),
        ):
            return flux2_image.Flux2ImagePipeline.from_pretrained(
                torch_dtype=torch.float32,
                device="cpu",
                model_configs=[],
                tokenizer_config=tokenizer_config,
            )

    def test_mistral_encoder_loads_multimodal_processor(self):
        processor = object()
        config = _TokenizerConfig()
        pool = _ModelPool(text_encoder=object())

        with (
            patch.object(
                flux2_image.AutoProcessor,
                "from_pretrained",
                return_value=processor,
            ) as load_processor,
            patch.object(
                flux2_image.AutoTokenizer, "from_pretrained"
            ) as load_tokenizer,
        ):
            pipe = self._load_pipeline(pool, config)

        self.assertTrue(config.downloaded)
        self.assertIs(pipe.tokenizer, processor)
        load_processor.assert_called_once_with(config.path)
        load_tokenizer.assert_not_called()

    def test_qwen3_encoder_keeps_tokenizer_path(self):
        tokenizer = object()
        config = _TokenizerConfig()
        pool = _ModelPool(text_encoder_qwen3=object())

        with (
            patch.object(
                flux2_image.AutoProcessor, "from_pretrained"
            ) as load_processor,
            patch.object(
                flux2_image.AutoTokenizer,
                "from_pretrained",
                return_value=tokenizer,
            ) as load_tokenizer,
        ):
            pipe = self._load_pipeline(pool, config)

        self.assertTrue(config.downloaded)
        self.assertIs(pipe.tokenizer, tokenizer)
        load_processor.assert_not_called()
        load_tokenizer.assert_called_once_with(config.path)


if __name__ == "__main__":
    unittest.main()
