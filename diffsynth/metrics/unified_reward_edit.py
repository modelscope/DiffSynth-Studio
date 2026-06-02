import torch
from transformers import AutoProcessor
from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.unified_reward_edit import UnifiedRewardEditModel
from .base import Metric
from transformers.utils import logging
logging.set_verbosity_error()


DEFAULT_UNIFIED_REWARD_TASK = "edit_pointwise_score"


class UnifiedRewardEditMetric(Metric):
    def __init__(self, model: UnifiedRewardEditModel):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig = ModelConfig(
            model_id="DiffSynth-Studio/ImageMetrics",
            origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/model-*.safetensors",
        ),
        processor_config: ModelConfig = ModelConfig(
            model_id="DiffSynth-Studio/ImageMetrics",
            origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/",
        ),
        torch_dtype: torch.dtype = None,
        device: torch.device = get_device_type(),
        task: str = DEFAULT_UNIFIED_REWARD_TASK,
        max_new_tokens: int = 256,
        processor_kwargs: dict = None,
        vram_limit: float = None,
    ):
        processor_kwargs = processor_kwargs or {}
        model_pool = cls.download_and_load_models(
            [model_config],
            torch_dtype=torch_dtype or torch.bfloat16,
            device=device,
            vram_limit=vram_limit,
        )
        model = model_pool.fetch_model("image_metrics_unified_reward_edit")
        if model is None:
            model = model_pool.fetch_model("joyai_image_text_encoder")
        if model is None:
            raise ValueError("Cannot find model: image_metrics_unified_reward_edit")
        if hasattr(model, "model"):
            model = model.model

        processor_config.download_if_necessary()
        processor = AutoProcessor.from_pretrained(processor_config.path, **processor_kwargs)
        model = UnifiedRewardEditModel(
            model=model,
            processor=processor,
            task=task,
            max_new_tokens=max_new_tokens,
        ).eval()
        return cls(model)


    @staticmethod
    def _primary_score(parsed: dict, task: str):
        if task == "edit_pairwise_rank":
            winner = parsed.get("winner")
            if isinstance(winner, (int, float)):
                return int(winner)
            if isinstance(winner, str):
                winner = winner.lower()
                if "equally" in winner or "tie" in winner or ("image 1" in winner and "image 2" in winner):
                    return 0
                if "image 1" in winner or "first image" in winner:
                    return 1
                if "image 2" in winner or "second image" in winner:
                    return 2
            return 0
        if task == "edit_pairwise_score":
            return [parsed.get("image_1_score"), parsed.get("image_2_score")]
        return parsed.get("score")


    @torch.no_grad()
    def evaluate(self, prompt: str | list[str] | None, images, task: str = DEFAULT_UNIFIED_REWARD_TASK):
        outputs = self.model(prompt, images, task=task)
        return [{**output, "score": self._primary_score(output, task)} for output in outputs]

    @torch.no_grad()
    def score(self, prompt: str | list[str] | None, images, task: str = DEFAULT_UNIFIED_REWARD_TASK):
        outputs = self.evaluate(prompt, images, task=task)
        return [output["score"] for output in outputs]

    def compute(self, prompt: str | list[str] | None, images, task: str = DEFAULT_UNIFIED_REWARD_TASK):
        return self.score(prompt, images, task=task)

    def forward(self, prompt: str | list[str] | None, images, task: str = DEFAULT_UNIFIED_REWARD_TASK):
        return self.score(prompt, images, task=task)
