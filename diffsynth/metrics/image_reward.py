import torch
from transformers import BertTokenizer
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.image_reward import ImageRewardModel
from .base import Metric

class ImageRewardMetric(Metric):
    def __init__(self, model: ImageRewardModel):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="ImageReward/model.safetensors"),
        tokenizer_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="ImageReward/"),
        torch_dtype: torch.dtype = None,
        device: torch.device = get_device_type(),
        max_length: int = 35,
        tokenizer_kwargs: dict = None,
        vram_limit: float = None,
    ):

        tokenizer_kwargs = tokenizer_kwargs or {}
        model_pool = cls.download_and_load_models([model_config], torch_dtype=torch_dtype, device=device, vram_limit=vram_limit)
        model = model_pool.fetch_model("image_metrics_image_reward")
        tokenizer_config.download_if_necessary()
        tokenizer = BertTokenizer.from_pretrained(tokenizer_config.path, **tokenizer_kwargs)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
        model.tokenizer = tokenizer
        model.max_length = max_length
        model.mlp = model.mlp.float()
        model = model.eval()
        return cls(model)

    @torch.no_grad()
    def score(self, prompt: str | list[str], images):
        scores = self.model(prompt, images)
        return self.tensor_to_list(scores)

    def compute(self, prompt: str | list[str], images):
        return self.score(prompt, images)

    def forward(self, prompt: str | list[str], images):
        return self.score(prompt, images)
