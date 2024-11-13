from .base_prompter import BasePrompter
from ..models.flux_text_encoder import FluxTextEncoder2
from ..models.sd3_text_encoder import SD3TextEncoder1
from transformers import CLIPTokenizer, T5TokenizerFast
import os, torch


class FluxPrompter(BasePrompter):
    def __init__(
        self,
        tokenizer_1_path=None,
        tokenizer_2_path=None
    ):
        if tokenizer_1_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_1_path = os.path.join(base_path, "tokenizer_configs/flux/tokenizer_1")
        if tokenizer_2_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_2_path = os.path.join(base_path, "tokenizer_configs/flux/tokenizer_2")
        super().__init__()
        self.tokenizer_1 = CLIPTokenizer.from_pretrained(tokenizer_1_path)
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(tokenizer_2_path)
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: FluxTextEncoder2 = None


    def fetch_models(self, text_encoder_1: SD3TextEncoder1 = None, text_encoder_2: FluxTextEncoder2 = None):
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2


    def encode_prompt_using_clip(self, prompt, text_encoder, tokenizer, max_length, device):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        ).input_ids.to(device)
        pooled_prompt_emb, _ = text_encoder(input_ids)
        return pooled_prompt_emb
    

    def encode_prompt_using_t5(self, prompt, text_encoder, tokenizer, max_length, device):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(device)
        prompt_emb = text_encoder(input_ids)
        return prompt_emb
    

    def encode_prompt(
        self,
        prompt,
        positive=True,
        device="cuda",
        t5_sequence_length=512,
    ):
        prompt = self.process_prompt(prompt, positive=positive)
        
        # CLIP
        pooled_prompt_emb = self.encode_prompt_using_clip(prompt, self.text_encoder_1, self.tokenizer_1, 77, device)
        
        # T5
        prompt_emb = self.encode_prompt_using_t5(prompt, self.text_encoder_2, self.tokenizer_2, t5_sequence_length, device)

        # text_ids
        text_ids = torch.zeros(prompt_emb.shape[0], prompt_emb.shape[1], 3).to(device=device, dtype=prompt_emb.dtype)

        return prompt_emb, pooled_prompt_emb, text_ids
