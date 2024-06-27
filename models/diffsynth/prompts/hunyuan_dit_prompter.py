from .utils import Prompter
from transformers import BertModel, T5EncoderModel, BertTokenizer, AutoTokenizer
import warnings, os


class HunyuanDiTPrompter(Prompter):
    def __init__(
        self,
        tokenizer_path=None,
        tokenizer_t5_path=None
    ):
        if tokenizer_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_path = os.path.join(base_path, "tokenizer_configs/hunyuan_dit/tokenizer")
        if tokenizer_t5_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_t5_path = os.path.join(base_path, "tokenizer_configs/hunyuan_dit/tokenizer_t5")
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tokenizer_t5 = AutoTokenizer.from_pretrained(tokenizer_t5_path)


    def encode_prompt_using_signle_model(self, prompt, text_encoder, tokenizer, max_length, clip_skip, device):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
            clip_skip=clip_skip
        )
        return prompt_embeds, attention_mask
    

    def encode_prompt(
        self,
        text_encoder: BertModel,
        text_encoder_t5: T5EncoderModel,
        prompt,
        clip_skip=1,
        clip_skip_2=1,
        positive=True,
        device="cuda"
    ):
        prompt = self.process_prompt(prompt, positive=positive)
        
        # CLIP
        prompt_emb, attention_mask = self.encode_prompt_using_signle_model(prompt, text_encoder, self.tokenizer, self.tokenizer.model_max_length, clip_skip, device)

        # T5
        prompt_emb_t5, attention_mask_t5 = self.encode_prompt_using_signle_model(prompt, text_encoder_t5, self.tokenizer_t5, self.tokenizer_t5.model_max_length, clip_skip_2, device)
        
        return prompt_emb, attention_mask, prompt_emb_t5, attention_mask_t5
