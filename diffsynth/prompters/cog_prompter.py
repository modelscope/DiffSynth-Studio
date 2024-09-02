from .base_prompter import BasePrompter
from ..models.flux_text_encoder import FluxTextEncoder2
from transformers import T5TokenizerFast
import os


class CogPrompter(BasePrompter):
    def __init__(
        self,
        tokenizer_path=None
    ):
        if tokenizer_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_path = os.path.join(base_path, "tokenizer_configs/cog/tokenizer")
        super().__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)
        self.text_encoder: FluxTextEncoder2 = None


    def fetch_models(self, text_encoder: FluxTextEncoder2 = None):
        self.text_encoder = text_encoder


    def encode_prompt_using_t5(self, prompt, text_encoder, tokenizer, max_length, device):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(device)
        prompt_emb = text_encoder(input_ids)
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0]*prompt_emb.shape[1], -1))

        return prompt_emb
    

    def encode_prompt(
        self,
        prompt,
        positive=True,
        device="cuda"
    ):
        prompt = self.process_prompt(prompt, positive=positive)
        prompt_emb = self.encode_prompt_using_t5(prompt, self.text_encoder, self.tokenizer, 226, device)
        return prompt_emb
