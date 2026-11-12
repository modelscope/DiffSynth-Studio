"""Exercise training prompt routing on CPU without model weights or downloads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image
import torch

from diffsynth.diffusion.base_pipeline import PipelineUnitRunner
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImageUnit_PromptEmbedder


TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "examples/minimax_h3_image/model_training/train.py"
spec = importlib.util.spec_from_file_location("minimax_h3_image_train", TRAIN_SCRIPT)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class MiniMaxH3ImageTrainingPromptTest(unittest.TestCase):
    def setUp(self):
        # Bypass weight loading, but use the real training input builder and unit runner.
        self.model = train.MiniMaxH3ImageTrainingModule.__new__(train.MiniMaxH3ImageTrainingModule)
        torch.nn.Module.__init__(self.model)
        self.model.use_gradient_checkpointing = True
        self.model.use_gradient_checkpointing_offload = False
        self.model.extra_inputs = []
        self.tokenized = []

        def tokenize(text, add_special_tokens=False):
            self.tokenized.append(text)
            return {"input_ids": list(text.encode("utf-8"))}

        def encode(input_ids, **kwargs):
            return input_ids[0, :, None].float()

        self.pipe = SimpleNamespace(
            tokenizer=tokenize, text_encoder=encode, device="cpu", torch_dtype=torch.float32,
        )
        self.image = Image.new("RGB", (64, 96))

    def embed(self, prompt):
        inputs = self.model.get_pipeline_inputs({"prompt": prompt, "image": self.image})
        return PipelineUnitRunner()(MiniMaxH3ImageUnit_PromptEmbedder(), self.pipe, *inputs)

    def test_prompt_is_in_positive_branch(self):
        shared, positive, _ = self.model.get_pipeline_inputs({"prompt": "a pagoda", "image": self.image})
        self.assertEqual(positive.get("prompt"), "a pagoda")
        self.assertNotIn("prompt", shared)
        self.assertIs(shared["input_image"], self.image)
        self.assertEqual((shared["width"], shared["height"]), self.image.size)

    def test_distinct_captions_reach_the_prompt_embedder(self):
        embeddings = []
        for prompt in ("a pagoda", "a statue"):
            _, positive, negative = self.embed(prompt)
            expected = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.float32)[:, None]
            self.assertTrue(torch.equal(positive["prompt_embeds"], expected))
            self.assertTrue(torch.equal(negative["prompt_embeds"], expected))  # cfg_scale == 1
            embeddings.append(positive["prompt_embeds"])
        self.assertEqual(self.tokenized, ["a pagoda", "a statue"])
        self.assertFalse(torch.equal(*embeddings))

    def test_empty_caption_keeps_existing_space_fallback(self):
        _, positive, _ = self.embed("")
        self.assertEqual(self.tokenized, [" "])
        self.assertEqual(positive["prompt_embeds"].shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
