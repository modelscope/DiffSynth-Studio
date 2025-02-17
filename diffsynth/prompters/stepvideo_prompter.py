from .base_prompter import BasePrompter
from ..models.hunyuan_dit_text_encoder import HunyuanDiTCLIPTextEncoder
from ..models.stepvideo_text_encoder import STEP1TextEncoder
from transformers import BertTokenizer
import os, torch


class StepVideoPrompter(BasePrompter):

    def __init__(
        self,
        tokenizer_1_path=None,
    ):
        if tokenizer_1_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_1_path = os.path.join(
                base_path, "tokenizer_configs/hunyuan_dit/tokenizer")
        super().__init__()
        self.tokenizer_1 = BertTokenizer.from_pretrained(tokenizer_1_path)

    def fetch_models(self, text_encoder_1: HunyuanDiTCLIPTextEncoder = None, text_encoder_2: STEP1TextEncoder = None):
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2

    def encode_prompt_using_clip(self, prompt, max_length, device):
        text_inputs = self.tokenizer_1(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder_1(
            text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )
        return prompt_embeds

    def encode_prompt_using_llm(self, prompt, max_length, device):
        y, y_mask = self.text_encoder_2(prompt, max_length=max_length, device=device)
        return y, y_mask

    def encode_prompt(self,
                      prompt,
                      positive=True,
                      device="cuda"):

        prompt = self.process_prompt(prompt, positive=positive)

        clip_embeds = self.encode_prompt_using_clip(prompt, max_length=77, device=device)
        llm_embeds, llm_mask = self.encode_prompt_using_llm(prompt, max_length=320, device=device)

        llm_mask = torch.nn.functional.pad(llm_mask, (clip_embeds.shape[1], 0), value=1)

        return clip_embeds, llm_embeds, llm_mask
