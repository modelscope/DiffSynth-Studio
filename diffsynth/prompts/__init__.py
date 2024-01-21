from transformers import CLIPTokenizer, AutoTokenizer
from ..models import SDTextEncoder, SDXLTextEncoder, SDXLTextEncoder2, ModelManager
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


class BeautifulPrompt:
    def __init__(self, tokenizer_path="configs/beautiful_prompt/tokenizer", model=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.template = 'Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {raw_prompt}\nOutput:'
    
    def __call__(self, raw_prompt):
        model_input = self.template.format(raw_prompt=raw_prompt)
        input_ids = self.tokenizer.encode(model_input, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=384,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1
        )
        prompt = raw_prompt + ", " + self.tokenizer.batch_decode(
            outputs[:, input_ids.size(1):],
            skip_special_tokens=True
        )[0].strip()
        return prompt
    

class Translator:
    def __init__(self, tokenizer_path="configs/translator/tokenizer", model=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model

    def __call__(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        output_ids = self.model.generate(input_ids)
        prompt = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return prompt
    

class Prompter:
    def __init__(self):
        self.tokenizer: CLIPTokenizer = None
        self.keyword_dict = {}
        self.translator: Translator = None
        self.beautiful_prompt: BeautifulPrompt = None

    def load_textual_inversion(self, textual_inversion_dict):
        self.keyword_dict = {}
        additional_tokens = []
        for keyword in textual_inversion_dict:
            tokens, _ = textual_inversion_dict[keyword]
            additional_tokens += tokens
            self.keyword_dict[keyword] = " " + " ".join(tokens) + " "
        self.tokenizer.add_tokens(additional_tokens)

    def load_beautiful_prompt(self, model, model_path):
        model_folder = os.path.dirname(model_path)
        self.beautiful_prompt = BeautifulPrompt(tokenizer_path=model_folder, model=model)
        if model_folder.endswith("v2"):
            self.beautiful_prompt.template = """Converts a simple image description into a prompt. \
Prompts are formatted as multiple related tags separated by commas, plus you can use () to increase the weight, [] to decrease the weight, \
or use a number to specify the weight. You should add appropriate words to make the images described in the prompt more aesthetically pleasing, \
but make sure there is a correlation between the input and output.\n\
### Input: {raw_prompt}\n### Output:"""

    def load_translator(self, model, model_path):
        model_folder = os.path.dirname(model_path)
        self.translator = Translator(tokenizer_path=model_folder, model=model)

    def load_from_model_manager(self, model_manager: ModelManager):
        self.load_textual_inversion(model_manager.textual_inversion_dict)
        if "translator" in model_manager.model:
            self.load_translator(model_manager.model["translator"], model_manager.model_path["translator"])
        if "beautiful_prompt" in model_manager.model:
            self.load_beautiful_prompt(model_manager.model["beautiful_prompt"], model_manager.model_path["beautiful_prompt"])

    def process_prompt(self, prompt, positive=True):
        for keyword in self.keyword_dict:
            if keyword in prompt:
                prompt = prompt.replace(keyword, self.keyword_dict[keyword])
        if positive and self.translator is not None:
            prompt = self.translator(prompt)
            print(f"Your prompt is translated: \"{prompt}\"")
        if positive and self.beautiful_prompt is not None:
            prompt = self.beautiful_prompt(prompt)
            print(f"Your prompt is refined by BeautifulPrompt: \"{prompt}\"")
        return prompt


class SDPrompter(Prompter):
    def __init__(self, tokenizer_path="configs/stable_diffusion/tokenizer"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

    def encode_prompt(self, text_encoder: SDTextEncoder, prompt, clip_skip=1, device="cuda", positive=True):
        prompt = self.process_prompt(prompt, positive=positive)
        input_ids = tokenize_long_prompt(self.tokenizer, prompt).to(device)
        prompt_emb = text_encoder(input_ids, clip_skip=clip_skip)
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0]*prompt_emb.shape[1], -1))

        return prompt_emb


class SDXLPrompter(Prompter):
    def __init__(
        self,
        tokenizer_path="configs/stable_diffusion/tokenizer",
        tokenizer_2_path="configs/stable_diffusion_xl/tokenizer_2"
    ):
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
