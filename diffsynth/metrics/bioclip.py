import torch
from transformers import CLIPTokenizer
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.bioclip import BioCLIPv2Compute
from .base import Metric


class BioCLIPMetric(Metric):
    def __init__(self, model: BioCLIPv2Compute):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="BioCLIPv2/open_clip_model.safetensors"),
        tokenizer_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="BioCLIPv2/"),
        torch_dtype: torch.dtype = None,
        device: torch.device = get_device_type(),
        max_length: int = 77,
        vram_limit: float = None,
    ):
        model_pool = cls.download_and_load_models([model_config], torch_dtype=torch_dtype, device=device, vram_limit=vram_limit)
        model = model_pool.fetch_model("image_metrics_bioclip_v2")
        tokenizer_config.download_if_necessary()
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_config.path)
        model = BioCLIPv2Compute(model=model, tokenizer=tokenizer, max_length=max_length).eval()
        return cls(model)

    @torch.no_grad()
    def score(self, prompt, images):
        scores = self.model(prompt, images)
        return self.tensor_to_list(scores)

    @torch.no_grad()
    def similarity_matrix(self, prompt, images):
        scores = self.model.similarity_matrix(prompt, images)
        return self.tensor_to_list(scores)

    def compute(self, prompt, images):
        return self.score(prompt, images)

    def forward(self, prompt, images):
        return self.score(prompt, images)
