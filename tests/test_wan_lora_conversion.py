import pytest
import torch

from diffsynth.utils.lora.wan import WanLoRAConverter


@pytest.mark.parametrize("module", WanLoRAConverter.modules)
@pytest.mark.parametrize("legacy", [False, True])
def test_round_trip_preserves_weights_and_delta(module, legacy):
    torch.manual_seed(0)
    suffix = ".default" if legacy else ""
    base = f"pipe.dit.blocks.2.{module}"
    a, b = torch.randn(3, 7), torch.randn(5, 3)
    source = {base + f".lora_A{suffix}.weight": a, base + f".lora_B{suffix}.weight": b}
    converter = WanLoRAConverter()
    exported = converter.align_to_opensource_format(source)
    restored = converter.align_to_diffsynth_format(exported)
    torch.testing.assert_close(
        restored[f"blocks.2.{module}.lora_A.weight"], a, rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored[f"blocks.2.{module}.lora_B.weight"] @ a, b @ a, rtol=0, atol=0
    )


def test_alpha_and_explicit_scale_are_applied_once():
    a, b = torch.randn(2, 4), torch.randn(3, 2)
    source = {
        "blocks.0.attn1.to_q.lora_A.weight": a,
        "blocks.0.attn1.to_q.lora_B.weight": b,
        "blocks.0.attn1.to_q.alpha": torch.tensor(6.0),
    }
    converted = WanLoRAConverter().align_to_diffsynth_format(source, scale=0.5)
    torch.testing.assert_close(
        converted["blocks.0.self_attn.q.lora_B.weight"] @ a, 1.5 * (b @ a)
    )


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"unknown": torch.ones(1)},
        {"blocks.0.self_attn.q.lora_A.weight": torch.ones(2, 4)},
        {
            "blocks.0.self_attn.q.lora_A.weight": torch.ones(2, 4),
            "blocks.0.self_attn.q.lora_B.weight": torch.ones(5, 3),
        },
    ],
)
def test_invalid_adapters_fail_explicitly(source):
    with pytest.raises(ValueError):
        WanLoRAConverter().align_to_opensource_format(source)


def test_export_names_agree_with_the_existing_base_weight_converter():
    from diffsynth.utils.state_dict_converters.wan_video_dit import (
        WanVideoDiTFromDiffusers,
    )

    for module in WanLoRAConverter.modules:
        a, b = torch.randn(2, 4), torch.randn(3, 2)
        exported = WanLoRAConverter.align_to_opensource_format(
            {
                f"blocks.3.{module}.lora_A.weight": a,
                f"blocks.3.{module}.lora_B.weight": b,
            }
        )
        key = next(key for key in exported if key.endswith(".lora_A.weight"))
        base_key = key.removeprefix("transformer.").replace(".lora_A", "")
        renamed = WanVideoDiTFromDiffusers({base_key: a})
        assert list(renamed) == [f"blocks.3.{module}.weight"]
        assert renamed[f"blocks.3.{module}.weight"] is a


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scaling_is_rejected(scale):
    source = {
        "blocks.0.self_attn.q.lora_A.weight": torch.ones(2, 4),
        "blocks.0.self_attn.q.lora_B.weight": torch.ones(3, 2),
    }
    with pytest.raises(ValueError, match="finite"):
        WanLoRAConverter.align_to_opensource_format(source, scale=scale)
    source["blocks.0.self_attn.q.alpha"] = torch.tensor(scale)
    with pytest.raises(ValueError, match="finite"):
        WanLoRAConverter.align_to_opensource_format(source)


def test_duplicate_normalized_factors_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        WanLoRAConverter.align_to_opensource_format(
            {
                "blocks.0.self_attn.q.lora_A.weight": torch.ones(2, 4),
                "pipe.dit.blocks.0.self_attn.q.lora_A.default.weight": torch.ones(2, 4),
            }
        )


def test_integer_factors_are_rejected():
    with pytest.raises(ValueError, match="floating-point"):
        WanLoRAConverter.align_to_opensource_format(
            {
                "blocks.0.self_attn.q.lora_A.weight": torch.ones(
                    2, 4, dtype=torch.int64
                ),
            }
        )


def test_cli_roundtrip_and_output_protection(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from safetensors.torch import load_file, save_file

    script = (
        Path(__file__).resolve().parents[1]
        / "examples/wanvideo/lora_conversion/convert.py"
    )
    source = {
        "blocks.0.self_attn.q.lora_A.weight": torch.randn(2, 4),
        "blocks.0.self_attn.q.lora_B.weight": torch.randn(3, 2),
    }
    original, exported, restored = [
        tmp_path / name
        for name in (
            "source.safetensors",
            "exported.safetensors",
            "restored.safetensors",
        )
    ]
    save_file(source, str(original))
    for input_path, output_path, target in [
        (original, exported, "diffusers"),
        (exported, restored, "diffsynth"),
    ]:
        subprocess.run(
            [
                sys.executable,
                str(script),
                str(input_path),
                str(output_path),
                "--to",
                target,
            ],
            check=True,
            capture_output=True,
        )
    actual = load_file(str(restored))
    assert actual.keys() == source.keys()
    for key in source:
        torch.testing.assert_close(actual[key], source[key], rtol=0, atol=0)
    for output_path in (original, exported):
        before = output_path.read_bytes()
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                str(original),
                str(output_path),
                "--to",
                "diffusers",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "Output already exists" in result.stderr
        assert output_path.read_bytes() == before
    invalid = tmp_path / "invalid.safetensors"
    save_file({"unsupported.weight": torch.ones(2, 2)}, str(invalid))
    missing = tmp_path / "not-created.safetensors"
    result = subprocess.run(
        [sys.executable, str(script), str(invalid), str(missing), "--to", "diffusers"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "Unsupported Wan LoRA key" in result.stderr
    assert not missing.exists()


def test_cli_rejects_an_output_created_during_conversion(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from safetensors.torch import save_file

    script = (
        Path(__file__).resolve().parents[1]
        / "examples/wanvideo/lora_conversion/convert.py"
    )
    spec = importlib.util.spec_from_file_location("wan_lora_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    save_file(
        {
            "blocks.0.self_attn.q.lora_A.weight": torch.ones(2, 4),
            "blocks.0.self_attn.q.lora_B.weight": torch.ones(3, 2),
        },
        str(source),
    )
    load = cli.load_file

    def create_output_before_loading(path):
        output.write_bytes(b"another writer's output")
        return load(path)

    monkeypatch.setattr(cli, "load_file", create_output_before_loading)
    monkeypatch.setattr(
        sys, "argv", [str(script), str(source), str(output), "--to", "diffusers"]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert output.read_bytes() == b"another writer's output"
