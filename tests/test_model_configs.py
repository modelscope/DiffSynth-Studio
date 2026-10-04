import pytest

from diffsynth.diffusion.training_module import DiffusionTrainingModule


@pytest.mark.parametrize("is_directory", [False, True])
def test_origin_paths_preserves_existing_local_path(tmp_path, is_directory):
    path = tmp_path / "checkpoint"
    if is_directory:
        path.mkdir()
    else:
        path.write_bytes(b"")
    module = DiffusionTrainingModule()

    configs = module.parse_model_configs(None, str(path))

    assert len(configs) == 1
    assert configs[0].path == str(path)
    assert configs[0].model_id is None
    configs[0].download_if_necessary()
    assert configs[0].path == str(path)


def test_origin_paths_preserves_remote_model_config():
    module = DiffusionTrainingModule()

    configs = module.parse_model_configs(None, "Wan-AI/Wan2.1-T2V-1.3B:*.safetensors")

    assert configs[0].path is None
    assert configs[0].model_id == "Wan-AI/Wan2.1-T2V-1.3B"
    assert configs[0].origin_file_pattern == "*.safetensors"
