from .base_prompter import BasePrompter, tokenize_long_prompt
from ..models.utils import load_state_dict, search_for_embeddings
from ..models import SDTextEncoder
from transformers import CLIPTokenizer
import torch, os



class SDPrompter(BasePrompter):
    def __init__(self, tokenizer_path=None):
        if tokenizer_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_path = os.path.join(base_path, "tokenizer_configs/stable_diffusion/tokenizer")
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.text_encoder: SDTextEncoder = None
        self.textual_inversion_dict = {}
        self.keyword_dict = {}


    def fetch_models(self, text_encoder: SDTextEncoder = None):
        self.text_encoder = text_encoder


    def add_textual_inversions_to_model(self, textual_inversion_dict, text_encoder):
        dtype = next(iter(text_encoder.parameters())).dtype
        state_dict = text_encoder.token_embedding.state_dict()
        token_embeddings = [state_dict["weight"]]
        for keyword in textual_inversion_dict:
            _, embeddings = textual_inversion_dict[keyword]
            token_embeddings.append(embeddings.to(dtype=dtype, device=token_embeddings[0].device))
        token_embeddings = torch.concat(token_embeddings, dim=0)
        state_dict["weight"] = token_embeddings
        text_encoder.token_embedding = torch.nn.Embedding(token_embeddings.shape[0], token_embeddings.shape[1])
        text_encoder.token_embedding = text_encoder.token_embedding.to(dtype=dtype, device=token_embeddings[0].device)
        text_encoder.token_embedding.load_state_dict(state_dict)


    def add_textual_inversions_to_tokenizer(self, textual_inversion_dict, tokenizer):
        additional_tokens = []
        for keyword in textual_inversion_dict:
            tokens, _ = textual_inversion_dict[keyword]
            additional_tokens += tokens
            self.keyword_dict[keyword] = " " + " ".join(tokens) + " "
        tokenizer.add_tokens(additional_tokens)


    def load_textual_inversions(self, model_paths):
        for model_path in model_paths:
            keyword = os.path.splitext(os.path.split(model_path)[-1])[0]
            state_dict = load_state_dict(model_path)

            # Search for embeddings
            for embeddings in search_for_embeddings(state_dict):
                if len(embeddings.shape) == 2 and embeddings.shape[1] == 768:
                    tokens = [f"{keyword}_{i}" for i in range(embeddings.shape[0])]
                    self.textual_inversion_dict[keyword] = (tokens, embeddings)

        self.add_textual_inversions_to_model(self.textual_inversion_dict, self.text_encoder)
        self.add_textual_inversions_to_tokenizer(self.textual_inversion_dict, self.tokenizer)


    def encode_prompt(self, prompt, clip_skip=1, device="cuda", positive=True):
        prompt = self.process_prompt(prompt, positive=positive)
        for keyword in self.keyword_dict:
            if keyword in prompt:
                print(f"Textual inversion {keyword} is enabled.")
                prompt = prompt.replace(keyword, self.keyword_dict[keyword])
        input_ids = tokenize_long_prompt(self.tokenizer, prompt).to(device)
        prompt_emb = self.text_encoder(input_ids, clip_skip=clip_skip)
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0]*prompt_emb.shape[1], -1))

        return prompt_emb