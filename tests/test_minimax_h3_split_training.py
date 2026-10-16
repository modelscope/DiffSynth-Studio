"""Real H3 split/cache/loss/PEFT updates with small deterministic model resources.

Only checkpoint loading, tokenization and the large text/video/audio encoders are
fixtures. The H3 DiT, dependency graph, cache pruning, loss and autograd are real.
"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from diffsynth.diffusion.base_pipeline import BasePipeline, PipelineUnit
from diffsynth.diffusion.training_module import (
    DiffusionTrainingModule,
    GeneralUnit_TemplateForward,
)
from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline
from examples.minimax_h3.model_training.train import MiniMaxH3TrainingModule

TEMPLATE_CODE = '''import torch
class EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32)
    def process_inputs(self, **kwargs):
        return {}
    def forward(self, **kwargs):
        return {"text_embedding": self.weight}
class DataProcessor:
    def __call__(self, **kwargs):
        return kwargs
TEMPLATE_MODEL = EmbeddingModel
TEMPLATE_DATA_PROCESSOR = DataProcessor
'''


class TokenizerFixture:
    def __call__(self, text, **kwargs):
        return {"input_ids": [1 + ord(char) % 7 for char in text]}


class TextEncoderFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 8)
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        return self.embed(input_ids)[0]


class VideoEncoderFixture(torch.nn.Module):
    def encode_video(self, frames, dtype=None, **kwargs):
        return frames.mean().expand(1, 24, 7, 2, 2).clone().to(dtype or frames.dtype)


class AudioEncoderFixture(torch.nn.Module):
    sample_rate = 32000

    def encode_audio(self, waveform, dtype=None):
        return waveform.mean().expand(2, 32, 37).clone().to(dtype or waveform.dtype)


def make_pipe(**kwargs):
    torch.manual_seed(311)
    pipe = MiniMaxH3Pipeline(device="cpu", torch_dtype=torch.float32)
    pipe.dit = MiniMaxH3DiT(
        num_layers=1, token_refiner_num_layers=0, hidden_size=32,
        num_attention_heads=1, attention_head_dim=128, ffn_hidden_size=64,
        text_dim=8, timestep_input_dim=16, time_embed_hidden_size=32,
        time_embed_dim=16, adaln_out_features=576, final_adaln_out_features=64,
    )
    pipe.text_encoder, pipe.tokenizer = TextEncoderFixture(), TokenizerFixture()
    pipe.video_vae, pipe.audio_vae = VideoEncoderFixture(), AudioEncoderFixture()
    return pipe


class H3SplitTrainingTests(unittest.TestCase):
    def setUp(self):
        self.rng_state, self.previous_threads = torch.get_rng_state(), torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_rng_state, self.rng_state)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.template = self.root / "template"
        self.template.mkdir()
        (self.template / "model.py").write_text(TEMPLATE_CODE)
        self.data = {
            "prompt": "a cat", "video": [Image.new("RGB", (32, 32), (64, 100, 180))] * 22,
            "input_audio": (torch.zeros(2, 29600), 32000), "template_inputs": {},
        }

    def make_model(self, task, owner="template_model", lora=True, cfg=1.0, checkpointing=False):
        template = str(self.template) if owner == "template_model" else None
        with patch.object(MiniMaxH3Pipeline, "from_pretrained", side_effect=make_pipe):
            return MiniMaxH3TrainingModule(
                trainable_models=owner, lora_base_model="dit" if lora else None,
                lora_target_modules="blocks.0.attn.qkv_proj,blocks.0.attn.out_proj", lora_rank=2,
                template_model_id_or_path=template, task=task, training_cfg_scale=cfg,
                extra_inputs="input_audio,template_inputs" if template else "input_audio",
                device="cpu", use_gradient_checkpointing=checkpointing,
            )

    def make_cache(self, **options):
        model = self.make_model("sft:data_process", **options)
        with torch.no_grad():
            cached = model(self.data)
        path = self.root / "cache.pth"
        torch.save(cached, path)
        return path, torch.load(path, weights_only=False)

    def updates(self, model, cache_path=None):
        optimizer = torch.optim.SGD(model.trainable_modules(), lr=0.1)
        history = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            cached = torch.load(cache_path, weights_only=False) if cache_path else None
            torch.manual_seed(91 + step)
            loss = model(self.data, inputs=cached)
            loss.backward()
            gradients = {
                name: None if value.grad is None else value.grad.detach().clone()
                for name, value in model.named_parameters() if value.requires_grad
            }
            optimizer.step()
            history.append((loss.detach(), gradients, copy.deepcopy(model.export_trainable_state_dict(model.state_dict()))))
        return history

    def assert_split_matches(self, owner="template_model", lora=True, cfg=1.0, prompt="a cat", checkpointing=False):
        self.data["prompt"] = prompt
        options = {"owner": owner, "lora": lora, "cfg": cfg, "checkpointing": checkpointing}
        full, split = self.make_model("sft", **options), self.make_model("sft:train", **options)
        initial = copy.deepcopy(split.export_trainable_state_dict(split.state_dict()))
        path, cached = self.make_cache(**options)
        expected, actual = self.updates(full), self.updates(split, path)
        for (expected_loss, expected_grads, expected_state), (loss, grads, state) in zip(expected, actual):
            if owner:
                selected = [value for name, value in grads.items() if f"pipe.{owner}." in name]
                self.assertTrue(any(value is not None and value.abs().sum() > 0 for value in selected))
            torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
            self.assertEqual(grads.keys(), expected_grads.keys())
            for name, value in grads.items():
                self.assertIsNotNone(value, name)
                torch.testing.assert_close(value, expected_grads[name], rtol=0, atol=0)
                torch.testing.assert_close(state[name], expected_state[name], rtol=0, atol=0)
        if owner:
            names = [name for name in initial if f"pipe.{owner}." in name]
            self.assertTrue(any(not torch.equal(initial[name], actual[-1][2][name]) for name in names))
            self.assertIs(cached[0]["video_latents"], cached[0]["input_latents"])
            self.assertIs(cached[0]["audio_latents"], cached[0]["audio_input_latents"])
        if lora:
            self.assertTrue(any("lora_B" in name and not torch.equal(initial[name], value) for name, value in actual[-1][2].items()))
        checkpoint = self.root / "trained.pth"
        torch.save(actual[-1][2], checkpoint)
        restored = self.make_model("sft:train", **options)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=False)
        for name, value in restored.export_trainable_state_dict(restored.state_dict()).items():
            torch.testing.assert_close(value, actual[-1][2][name], rtol=0, atol=0)
        expected_calls = 0 if prompt is None and cfg == 1 else 2 * (1 if cfg == 1 else 2)
        if owner:
            self.assertEqual(split.pipe.text_encoder.calls, expected_calls)
        else:
            self.assertEqual(split.pipe.text_encoder.calls, 0)
        return cached

    def test_template_and_lora_update_with_and_without_cfg(self):
        for cfg in (1.0, 2.0):
            with self.subTest(cfg=cfg):
                self.assert_split_matches(cfg=cfg)

    def test_template_and_lora_cross_gradient_checkpoint_with_cfg(self):
        self.assert_split_matches(cfg=2.0, checkpointing=True)

    def test_template_only_updates_including_explicit_none_prompt(self):
        for prompt in ("a cat", None):
            with self.subTest(prompt=prompt):
                self.assert_split_matches(lora=False, prompt=prompt)

    def test_text_encoder_updates_with_and_without_cfg(self):
        for cfg in (1.0, 2.0):
            with self.subTest(cfg=cfg):
                self.assert_split_matches(owner="text_encoder", cfg=cfg)

    def test_dit_only_preserves_old_cache_keys_and_updates(self):
        cached = self.assert_split_matches(owner=None)
        self.assertEqual(set(cached[0]), {
            "input_latents", "audio_input_latents", "cfg_scale", "imgvid_cond_noise_aug",
            "audio_cond_noise_aug", "use_gradient_checkpointing", "use_gradient_checkpointing_offload",
        })
        self.assertEqual(set(cached[1]), {"prompt_embeds", "packed"})
        self.assertEqual(cached[2], {})

    def test_deferred_vae_producers_only_keep_external_latents(self):
        for owner, expected in (("video_vae", {"audio_latents"}), ("audio_vae", {"video_latents"}),
                                ("video_vae,audio_vae", set())):
            with self.subTest(owner=owner):
                _, cached = self.make_cache(owner=owner)
                self.assertEqual({"video_latents", "audio_latents"} & cached[0].keys(), expected)
                loss = self.make_model("sft:train", owner=owner)({}, inputs=cached)
                loss.backward()

    def test_old_conditioning_caches_require_rebuilding(self):
        for owner in ("template_model", "text_encoder"):
            with self.subTest(owner=owner):
                _, cached = self.make_cache(owner=owner)
                cached[0].pop("video_latents", None)
                cached[0].pop("audio_latents", None)
                if owner == "template_model":
                    cached[1].pop("prompt", None)
                model = self.make_model("sft:train", owner=owner)
                with self.assertRaisesRegex(ValueError, "Rebuild.*sft:data_process"):
                    model({}, inputs=cached)

    def test_missing_prompt_is_not_silently_template_only(self):
        _, cached = self.make_cache()
        cached[1].pop("prompt", None)
        with self.assertRaisesRegex(ValueError, "prompt.*Rebuild"):
            self.make_model("sft:train")({}, inputs=cached)

    def test_cfg_cache_requires_negative_prompt(self):
        _, cached = self.make_cache(cfg=2.0)
        cached[2].pop("negative_prompt", None)
        with self.assertRaisesRegex(ValueError, "negative_prompt.*Rebuild"):
            self.make_model("sft:train", cfg=2.0)({}, inputs=cached)


class SplitBoundaryTests(unittest.TestCase):
    def prune(self, units, **options):
        module = DiffusionTrainingModule()
        module.pipe = BasePipeline(device="cpu", torch_dtype=torch.float32)
        module.pipe.units = units
        module.pipe.model_fn = lambda soft=None, ordinary=None: None
        module.split_pipeline_units("sft:data_process", module.pipe, trainable_models="train",
                                    remove_unnecessary_params=True, loss_required_params=(), **options)
        return module.pipe.units[-1].process(None, {"soft": 1, "ordinary": 2, "raw": 3}, {}, {})[0]

    def test_soft_keys_only_cross_boundary_before_their_first_deferred_producer(self):
        producer = PipelineUnit(input_params=("raw",), output_params=("soft", "ordinary"), onload_model_names=("train",))
        consumer = PipelineUnit(input_params=("soft", "ordinary"), output_params=("result",), onload_model_names=("train",))
        self.assertEqual(self.prune([producer, consumer], loss_recomputed_params=("soft",)), {"ordinary": 2, "raw": 3})
        self.assertEqual(self.prune([consumer, producer], loss_recomputed_params=("soft",)), {"soft": 1, "ordinary": 2, "raw": 3})
        self.assertEqual(self.prune([producer, consumer]), {"soft": 1, "ordinary": 2, "raw": 3})
        self.assertEqual(self.prune([consumer], loss_recomputed_params=("soft",), force_remove_params_shared=("soft",)), {"ordinary": 2})

    def test_existing_kv_template_dependency_stays_deferred(self):
        class KVTemplate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, **kwargs):
                return {"kv_cache": self.weight}

        class Consumer(PipelineUnit):
            def __init__(self):
                super().__init__(input_params=("kv_cache",), output_params=("result",))

            def process(self, pipe, kv_cache):
                return {"result": kv_cache.square()}

        module = DiffusionTrainingModule()
        module.pipe = BasePipeline(device="cpu", torch_dtype=torch.float32)
        module.pipe.template_model = KVTemplate()
        module.pipe.units = [GeneralUnit_TemplateForward(), Consumer()]
        module.split_pipeline_units("sft:train", module.pipe, trainable_models="template_model")
        self.assertEqual(len(module.pipe.units), 2)
        inputs = ({"template_inputs": {}}, {}, {})
        for unit in module.pipe.units:
            inputs = module.pipe.unit_runner(unit, module.pipe, *inputs)
        inputs[0]["result"].backward()
        self.assertEqual(module.pipe.template_model.weight.grad.item(), 4.0)


if __name__ == "__main__":
    unittest.main()
