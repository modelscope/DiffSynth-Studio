from transformers import CLIPTokenizer, AutoTokenizer
from ..models import ModelManager
import os


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
    def __init__(self, tokenizer_path=None, model=None):
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
    def __init__(self, tokenizer_path=None, model=None):
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
