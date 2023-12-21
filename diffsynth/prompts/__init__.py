from transformers import CLIPTokenizer
from ..models import SDTextEncoder, SDXLTextEncoder, SDXLTextEncoder2, load_state_dict
import torch, os


def tokenize_long_prompt(tokenizer, prompt):
    # Get model_max_length from self.tokenizer
    length = tokenizer.model_max_length

    # To avoid the warning. set self.tokenizer.model_max_length to +oo.
    tokenizer.model_max_length = 99999999

    # Tokenize it!
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Determine the real length.
    max_length = (input_ids.shape[1] + length - 1) // length * length

    # Restore tokenizer.model_max_length
    tokenizer.model_max_length = length
    
    # Tokenize it again with fixed length.
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    ).input_ids

    # Reshape input_ids to fit the text encoder.
    num_sentence = input_ids.shape[1] // length
    input_ids = input_ids.reshape((num_sentence, length))
    
    return input_ids


def search_for_embeddings(state_dict):
    embeddings = []
    for k in state_dict:
        if isinstance(state_dict[k], torch.Tensor):
            embeddings.append(state_dict[k])
        elif isinstance(state_dict[k], dict):
            embeddings += search_for_embeddings(state_dict[k])
    return embeddings


class SDPrompter:
    def __init__(self, tokenizer_path="configs/stable_diffusion/tokenizer"):
        # We use the tokenizer implemented by transformers
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.keyword_dict = {}
    
    def encode_prompt(self, text_encoder: SDTextEncoder, prompt, clip_skip=1, device="cuda"):
        for keyword in self.keyword_dict:
            if keyword in prompt:
                prompt = prompt.replace(keyword, self.keyword_dict[keyword])
        input_ids = tokenize_long_prompt(self.tokenizer, prompt).to(device)
        prompt_emb = text_encoder(input_ids, clip_skip=clip_skip)
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0]*prompt_emb.shape[1], -1))

        return prompt_emb
    
    def load_textual_inversion(self, textual_inversion_dict):
        self.keyword_dict = {}
        additional_tokens = []
        for keyword in textual_inversion_dict:
            tokens, _ = textual_inversion_dict[keyword]
            additional_tokens += tokens
            self.keyword_dict[keyword] = " " + " ".join(tokens) + " "
        self.tokenizer.add_tokens(additional_tokens)


class SDXLPrompter:
    def __init__(
        self,
        tokenizer_path="configs/stable_diffusion/tokenizer",
        tokenizer_2_path="configs/stable_diffusion_xl/tokenizer_2"
    ):
        # We use the tokenizer implemented by transformers
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_2_path)
    
    def encode_prompt(
        self,
        text_encoder: SDXLTextEncoder,
        text_encoder_2: SDXLTextEncoder2,
        prompt,
        clip_skip=1,
        clip_skip_2=2,
        device="cuda"
    ):
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
