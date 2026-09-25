import os
import pytest
from diffsynth.core.loader.config import ModelConfig


def make_files(root, names):
    for name in names:
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").close()


def resolve(tmp_path, origin_file_pattern):
    config = ModelConfig(
        model_id="org/model",
        origin_file_pattern=origin_file_pattern,
        local_model_path=str(tmp_path),
        skip_download=True,
    )
    config.download_if_necessary()
    return config.path


def test_incomplete_sharded_files_raise(tmp_path, monkeypatch):
    monkeypatch.delenv("DIFFSYNTH_MODEL_BASE_PATH", raising=False)
    shards = [f"transformer/diffusion_pytorch_model-{i:05d}-of-00009.safetensors" for i in (1, 3, 4, 7, 9)]
    make_files(tmp_path / "org" / "model", shards)
    with pytest.raises(ValueError, match="Incomplete sharded model files") as exc_info:
        resolve(tmp_path, "transformer/diffusion_pytorch_model*.safetensors")
    for i in (2, 5, 6, 8):
        assert f"diffusion_pytorch_model-{i:05d}-of-00009.safetensors" in str(exc_info.value)
    assert "diffusion_pytorch_model-00001-of-00009.safetensors" not in str(exc_info.value)


def test_single_shard_of_many_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("DIFFSYNTH_MODEL_BASE_PATH", raising=False)
    make_files(tmp_path / "org" / "model", ["model-00001-of-00002.safetensors"])
    with pytest.raises(ValueError, match="model-00002-of-00002.safetensors"):
        resolve(tmp_path, "model*.safetensors")


def test_complete_sharded_files_resolve(tmp_path, monkeypatch):
    monkeypatch.delenv("DIFFSYNTH_MODEL_BASE_PATH", raising=False)
    shards = [f"model-{i:05d}-of-00003.safetensors" for i in (1, 2, 3)]
    make_files(tmp_path / "org" / "model", shards + ["config.json"])
    path = resolve(tmp_path, "model*.safetensors")
    assert sorted(os.path.basename(p) for p in path) == shards


def test_non_sharded_file_resolves(tmp_path, monkeypatch):
    monkeypatch.delenv("DIFFSYNTH_MODEL_BASE_PATH", raising=False)
    make_files(tmp_path / "org" / "model", ["vae/diffusion_pytorch_model.safetensors"])
    path = resolve(tmp_path, "vae/diffusion_pytorch_model.safetensors")
    assert os.path.normpath(path) == os.path.join(str(tmp_path), "org", "model", "vae", "diffusion_pytorch_model.safetensors")
