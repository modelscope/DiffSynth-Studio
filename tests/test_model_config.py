import os, tempfile, unittest
from diffsynth import ModelConfig


class ModelConfigLocalFileTest(unittest.TestCase):
    """`ModelConfig.download_if_necessary` resolves `model_id` + `origin_file_pattern` to local files."""

    def setUp(self):
        self.model_base_path = tempfile.mkdtemp()
        self.environ_backup = {key: os.environ.get(key) for key in ["DIFFSYNTH_SKIP_DOWNLOAD", "DIFFSYNTH_MODEL_BASE_PATH"]}
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = self.model_base_path

    def tearDown(self):
        for key, value in self.environ_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_no_file_matches(self):
        model_config = ModelConfig(model_id="Example/Model", origin_file_pattern="transformer/*.safetensors")
        with self.assertRaises(ValueError):
            model_config.download_if_necessary()

    def test_file_matches(self):
        folder = os.path.join(self.model_base_path, "Example/Model", "transformer")
        os.makedirs(folder)
        file_path = os.path.join(folder, "model.safetensors")
        open(file_path, "w").close()
        model_config = ModelConfig(model_id="Example/Model", origin_file_pattern="transformer/*.safetensors")
        model_config.download_if_necessary()
        self.assertEqual(model_config.path, file_path)


if __name__ == "__main__":
    unittest.main()
