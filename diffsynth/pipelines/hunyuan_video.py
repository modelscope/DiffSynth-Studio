from ..models import ModelManager, SD3TextEncoder1
from .base import BasePipeline
from ..prompters import HunyuanVideoPrompter
import torch
from transformers import LlamaModel
from tqdm import tqdm

class HunyuanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        # 参照diffsynth的排序，text_encoder_1指CLIP；text_encoder_2指llm，与hunyuanvideo源代码刚好相反
        self.prompter = HunyuanVideoPrompter()
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: LlamaModel = None
    

    def fetch_models(self, model_manager: ModelManager):
        self.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("hunyuan_video_text_encoder_2")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)

    @staticmethod
    def from_model_manager(model_manager: ModelManager, device=None):

        pipe = HunyuanVideoPipeline(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_models(model_manager)
        return pipe

    def encode_prompt(self, prompt, positive=True, clip_sequence_length=77, llm_sequence_length=256):
        prompt_emb, pooled_prompt_emb = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive, clip_sequence_length=clip_sequence_length, llm_sequence_length=llm_sequence_length
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb}

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        seed=None,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        pass

        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        return prompt_emb_posi