from .base_prompter import BasePrompter
from ..models.model_manager import ModelManager
from ..models import SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3
from transformers import CLIPTokenizer, T5TokenizerFast
import os, torch


class SD3Prompter(BasePrompter):
    def __init__(
        self,
        tokenizer_1_path=None,
        tokenizer_2_path=None,
        tokenizer_3_path=None
    ):
        if tokenizer_1_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_1_path = os.path.join(base_path, "tokenizer_configs/stable_diffusion_3/tokenizer_1")
        if tokenizer_2_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_2_path = os.path.join(base_path, "tokenizer_configs/stable_diffusion_3/tokenizer_2")
        if tokenizer_3_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_3_path = os.path.join(base_path, "tokenizer_configs/stable_diffusion_3/tokenizer_3")
        super().__init__()
        self.tokenizer_1 = CLIPTokenizer.from_pretrained(tokenizer_1_path)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_2_path)
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(tokenizer_3_path)
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: SD3TextEncoder2 = None
        self.text_encoder_3: SD3TextEncoder3 = None


    def fetch_models(self, text_encoder_1: SD3TextEncoder1 = None, text_encoder_2: SD3TextEncoder2 = None, text_encoder_3: SD3TextEncoder3 = None):
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3


    def encode_prompt_using_clip(self, prompt, text_encoder, tokenizer, max_length, device):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        ).input_ids.to(device)
        pooled_prompt_emb, prompt_emb = text_encoder(input_ids)
        return pooled_prompt_emb, prompt_emb
    

    def encode_prompt_using_t5(self, prompt, text_encoder, tokenizer, max_length, device):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        ).input_ids.to(device)
        prompt_emb = text_encoder(input_ids)
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0]*prompt_emb.shape[1], -1))

        return prompt_emb
    

    def encode_prompt(
        self,
        prompt,
        positive=True,
        device="cuda",
        t5_sequence_length=77,
    ):
        prompt = self.process_prompt(prompt, positive=positive)
        
        # CLIP
        pooled_prompt_emb_1, prompt_emb_1 = self.encode_prompt_using_clip(prompt, self.text_encoder_1, self.tokenizer_1, 77, device)
        pooled_prompt_emb_2, prompt_emb_2 = self.encode_prompt_using_clip(prompt, self.text_encoder_2, self.tokenizer_2, 77, device)

        # T5
        if self.text_encoder_3 is None:
            prompt_emb_3 = torch.zeros((prompt_emb_1.shape[0], t5_sequence_length, 4096), dtype=prompt_emb_1.dtype, device=device)
        else:
            prompt_emb_3 = self.encode_prompt_using_t5(prompt, self.text_encoder_3, self.tokenizer_3, t5_sequence_length, device)
            prompt_emb_3 = prompt_emb_3.to(prompt_emb_1.dtype) # float32 -> float16

        # Merge
        prompt_emb = torch.cat([
            torch.nn.functional.pad(torch.cat([prompt_emb_1, prompt_emb_2], dim=-1), (0, 4096 - 768 - 1280)),
            prompt_emb_3
        ], dim=-2)
        pooled_prompt_emb = torch.cat([pooled_prompt_emb_1, pooled_prompt_emb_2], dim=-1)

        return prompt_emb, pooled_prompt_emb
