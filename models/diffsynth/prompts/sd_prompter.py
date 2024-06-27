from .utils import Prompter, tokenize_long_prompt
from transformers import CLIPTokenizer
from ..models import SDTextEncoder
import os


class SDPrompter(Prompter):
    def __init__(self, tokenizer_path=None):
        if tokenizer_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_path = os.path.join(base_path, "tokenizer_configs/stable_diffusion/tokenizer")
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

    def encode_prompt(self, text_encoder: SDTextEncoder, prompt, clip_skip=1, device="cuda", positive=True):
        prompt = self.process_prompt(prompt, positive=positive)
        input_ids = tokenize_long_prompt(self.tokenizer, prompt).to(device)
        prompt_emb = text_encoder(input_ids, clip_skip=clip_skip)
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0]*prompt_emb.shape[1], -1))

        return prompt_emb