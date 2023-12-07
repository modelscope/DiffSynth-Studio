from transformers import CLIPTokenizer
from .sd_tokenizer import SDTokenizer


class SDXLTokenizer(SDTokenizer):
    def __init__(self):
        super().__init__()


class SDXLTokenizer2:
    def __init__(self, tokenizer_path="configs/stable_diffusion_xl/tokenizer_2"):
        # We use the tokenizer implemented by transformers
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    
    def __call__(self, prompt):
        # Get model_max_length from self.tokenizer
        length = self.tokenizer.model_max_length

        # To avoid the warning. set self.tokenizer.model_max_length to +oo.
        self.tokenizer.model_max_length = 99999999

        # Tokenize it!
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        # Determine the real length.
        max_length = (input_ids.shape[1] + length - 1) // length * length

        # Restore self.tokenizer.model_max_length
        self.tokenizer.model_max_length = length
        
        # Tokenize it again with fixed length.
        input_ids = self.tokenizer(
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

