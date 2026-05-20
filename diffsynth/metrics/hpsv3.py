from transformers import AutoProcessor
import torch
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.hpsv3 import HPSv3Model
from .base import Metric


class HPSv3Metric(Metric):
    def __init__(self, model: HPSv3Model):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="HPSv3/model.safetensors"),
        processor_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="HPSv3/"),
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = get_device_type(),
        score_index: int = 0,
        use_special_tokens: bool = True,
        max_pixels: int = 256 * 28 * 28,
        min_pixels: int = 256 * 28 * 28,
        processor_kwargs: dict = None,
        vram_limit: float = None,
    ):
        
        processor_kwargs = processor_kwargs or {}
        model_pool = cls.download_and_load_models([model_config], torch_dtype=torch_dtype, device=device, vram_limit=vram_limit)
        model = model_pool.fetch_model("image_metrics_hpsv3")
        processor_config.download_if_necessary()
        processor = AutoProcessor.from_pretrained(processor_config.path, padding_side="right", **processor_kwargs)
        if use_special_tokens:
            special_tokens = ["<|Reward|>"]
            processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            model.special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)
            model.reward_token = "special"
        model.config.tokenizer_padding_side = processor.tokenizer.padding_side
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        if hasattr(model.config, "text_config"):
            model.config.text_config.pad_token_id = processor.tokenizer.pad_token_id
        model.rm_head.to(torch.float32)
        model = HPSv3Model(
            model=model,
            processor=processor,
            use_special_tokens=use_special_tokens,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            score_index=score_index,
        ).eval()
        return cls(model)

    @torch.no_grad()
    def score(self, prompt: str | list[str], images):
        scores = self.model(prompt, images)
        return self.tensor_to_list(scores)

    def compute(self, prompt: str | list[str], images):
        return self.score(prompt, images)

    def forward(self, prompt: str | list[str], images):
        return self.score(prompt, images)
