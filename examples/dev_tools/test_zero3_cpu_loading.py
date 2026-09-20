"""CPU unit tests; DeepSpeed loading is mocked, not a distributed GPU test."""
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.target_device = None

    def to(self, *, dtype, device):
        self.target_device = device
        return super().to(dtype=dtype)


class LoadingTest(unittest.TestCase):
    def run_load(self, zero3, enabled, disk, supplied=False, converter=False):
        reads, inits, zero3_loads = [], [], []
        expected = {"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}

        def read(path, device, torch_dtype=None):
            reads.append(device)
            return {k: v.to(torch_dtype) for k, v in expected.items()}

        def init(**kwargs):
            inits.append(kwargs)
            return []

        def load_zero(model, state):
            zero3_loads.append(state)
            self.assertTrue(all(t.device.type == "cpu" for t in state.values()))
            model.load_state_dict(state)

        module = ModuleType("transformers.integrations.deepspeed")
        module._load_state_dict_into_zero3_model = load_zero
        ns = definitions("diffsynth/core/loader/model.py", ["load_model"], {
            "torch": torch, "is_deepspeed_zero3_enabled": lambda: zero3,
            "get_init_context": init, "ContextManagers": lambda _: nullcontext(),
            "DiskMap": read, "load_state_dict": lambda path, dtype, device: read(path, device, dtype),
        })
        with patch.dict(sys.modules, {module.__name__: module}):
            model = ns["load_model"](
                TinyModel, "mock.safetensors", torch_dtype=torch.float32, device="cuda:3",
                use_disk_map=disk, zero3_load_state_dict_on_cpu=enabled,
                state_dict=expected if supplied else None,
                state_dict_converter=(lambda state: dict(state)) if converter else None,
            )
        self.assertEqual(reads, [] if supplied else ["cpu" if zero3 and enabled else "cuda:3"])
        self.assertEqual(inits[0]["device"], "cuda:3")
        self.assertEqual(model.target_device, "cuda:3")
        torch.testing.assert_close(model.weight, expected["weight"])
        self.assertEqual(len(zero3_loads), int(zero3))
        if supplied and zero3 and converter:
            self.assertIs(zero3_loads[0]["weight"], expected["weight"])

    def test_loading_modes_and_default_behavior(self):
        for zero3 in (False, True):
            for enabled in (False, True):
                for disk in (False, True):
                    for converter in (False, True):
                        with self.subTest(zero3=zero3, enabled=enabled, disk=disk, converter=converter):
                            self.run_load(zero3, enabled, disk, converter=converter)

    def test_supplied_state_dict_is_not_reloaded(self):
        self.run_load(True, True, True, supplied=True, converter=True)

    def test_unsupported_paths_fail_before_model_creation(self):
        ns = definitions("diffsynth/core/loader/model.py", ["load_model"], {
            "torch": torch, "is_deepspeed_zero3_enabled": lambda: True,
        })
        for extra in ({"module_map": {}}, {"quantize": object()}):
            with self.assertRaisesRegex(ValueError, "standard, non-quantized"):
                ns["load_model"](TinyModel, "unused", zero3_load_state_dict_on_cpu=True, **extra)

    def test_parameter_reaches_loader_through_model_pool(self):
        calls = []
        registry = [{"model_hash": "tiny", "model_class": "TinyModel", "model_name": "tiny"}]
        ns = definitions("diffsynth/models/model_loader.py", ["ModelPool"], {
            "torch": torch, "json": __import__("json"), "MODEL_CONFIGS": registry,
            "hash_model_file": lambda path: "tiny", "load_model": lambda *args, **kwargs: calls.append(kwargs) or TinyModel(),
        })
        pool = ns["ModelPool"]()
        pool.import_model_class = lambda name: TinyModel
        pool.fetch_module_map = lambda *args: None
        pool.auto_load_model("unused", zero3_load_state_dict_on_cpu=True)
        self.assertTrue(calls[0]["zero3_load_state_dict_on_cpu"])

    def test_cli_and_pipeline_forwarding(self):
        import argparse
        path = "examples/minimax_h3/model_training/train.py"
        ns = definitions(path, ["minimax_h3_parser"], {
            "argparse": argparse, "add_general_config": lambda p: p, "add_video_size_config": lambda p: p,
        })
        parser = ns["minimax_h3_parser"]()
        self.assertFalse(parser.parse_args([]).zero3_load_state_dict_on_cpu)
        self.assertTrue(parser.parse_args(["--zero3_load_state_dict_on_cpu"]).zero3_load_state_dict_on_cpu)
        tree = ast.parse((ROOT / "diffsynth/diffusion/base_pipeline.py").read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "auto_load_model"]
        self.assertTrue(any(any(k.arg == "zero3_load_state_dict_on_cpu" for k in n.keywords) for n in calls))


if __name__ == "__main__":
    unittest.main()
