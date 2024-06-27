from .utils import Prompter, tokenize_long_prompt
from transformers import CLIPTokenizer
from ..models import SDXLTextEncoder, SDXLTextEncoder2
import torch, os


class SDXLPrompter(Prompter):
    def __init__(
        self,
        tokenizer_path=None,
        tokenizer_2_path=None
    ):
        if tokenizer_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_path = os.path.join(base_path, "tokenizer_configs/stable_diffusion/tokenizer")
        if tokenizer_2_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_2_path = os.path.join(base_path, "tokenizer_configs/stable_diffusion_xl/tokenizer_2")
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_2_path)
    
    def encode_prompt(
        self,
        text_encoder: SDXLTextEncoder,
        text_encoder_2: SDXLTextEncoder2,
        prompt,
        clip_skip=1,
        clip_skip_2=2,
        positive=True,
        device="cuda"
    ):
        prompt = self.process_prompt(prompt, positive=positive)
        
        # 1
        input_ids = tokenize_long_prompt(self.tokenizer, prompt).to(device)
        prompt_emb_1 = text_encoder(input_ids, clip_skip=clip_skip)

        # 2
        input_ids_2 = tokenize_long_prompt(self.tokenizer_2, prompt).to(device)
        add_text_embeds, prompt_emb_2 = text_encoder_2(input_ids_2, clip_skip=clip_skip_2)

        # Merge
        prompt_emb = torch.concatenate([prompt_emb_1, prompt_emb_2], dim=-1)

        # For very long prompt, we only use the first 77 tokens to compute `add_text_embeds`.
        add_text_embeds = add_text_embeds[0:1]
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0]*prompt_emb.shape[1], -1))
        return add_text_embeds, prompt_emb
