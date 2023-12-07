from transformers import CLIPTokenizer
from ..models import SDTextEncoder, SDXLTextEncoder, SDXLTextEncoder2
import torch, os
from safetensors import safe_open


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


def load_textual_inversion(prompt):
    # TODO: This module is not enabled now.
    textual_inversion_files = os.listdir("models/textual_inversion")
    embeddings_768 = []
    embeddings_1280 = []
    for file_name in textual_inversion_files:
        if not file_name.endswith(".safetensors"):
            continue
        keyword = file_name[:-len(".safetensors")]
        if keyword in prompt:
            prompt = prompt.replace(keyword, "")
            with safe_open(f"models/textual_inversion/{file_name}", framework="pt", device="cpu") as f:
                for k in f.keys():
                    embedding = f.get_tensor(k).to(torch.float32)
                if embedding.shape[-1] == 768:
                    embeddings_768.append(embedding)
                elif embedding.shape[-1] == 1280:
                    embeddings_1280.append(embedding)

    if len(embeddings_768)==0:
        embeddings_768 = torch.zeros((0, 768))
    else:
        embeddings_768 = torch.concat(embeddings_768, dim=0)

    if len(embeddings_1280)==0:
        embeddings_1280 = torch.zeros((0, 1280))
    else:
        embeddings_1280 = torch.concat(embeddings_1280, dim=0)

    return prompt, embeddings_768, embeddings_1280


class SDPrompter:
    def __init__(self, tokenizer_path="configs/stable_diffusion/tokenizer"):
        # We use the tokenizer implemented by transformers
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    
    def encode_prompt(self, text_encoder: SDTextEncoder, prompt, clip_skip=1, device="cuda"):
        input_ids = tokenize_long_prompt(self.tokenizer, prompt).to(device)
        prompt_emb = text_encoder(input_ids, clip_skip=clip_skip)
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0]*prompt_emb.shape[1], -1))

        return prompt_emb


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
