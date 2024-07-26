from transformers import AutoTokenizer
from ..models.model_manager import ModelManager
import torch



class BeautifulPrompt(torch.nn.Module):
    def __init__(self, tokenizer_path=None, model=None, template=""):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.template = template


    @staticmethod
    def from_model_manager(model_nameger: ModelManager):
        model, model_path = model_nameger.fetch_model("beautiful_prompt", require_model_path=True)
        template = 'Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {raw_prompt}\nOutput:'
        if model_path.endswith("v2"):
            template = """Converts a simple image description into a prompt. \
Prompts are formatted as multiple related tags separated by commas, plus you can use () to increase the weight, [] to decrease the weight, \
or use a number to specify the weight. You should add appropriate words to make the images described in the prompt more aesthetically pleasing, \
but make sure there is a correlation between the input and output.\n\
### Input: {raw_prompt}\n### Output:"""
        beautiful_prompt = BeautifulPrompt(
            tokenizer_path=model_path,
            model=model,
            template=template
        )
        return beautiful_prompt
    

    def __call__(self, raw_prompt, positive=True, **kwargs):
        if positive:
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
            print(f"Your prompt is refined by BeautifulPrompt: {prompt}")
            return prompt
        else:
            return raw_prompt
    


class Translator(torch.nn.Module):
    def __init__(self, tokenizer_path=None, model=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model


    @staticmethod
    def from_model_manager(model_nameger: ModelManager):
        model, model_path = model_nameger.fetch_model("translator", require_model_path=True)
        translator = Translator(tokenizer_path=model_path, model=model)
        return translator
    

    def __call__(self, prompt, **kwargs):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        output_ids = self.model.generate(input_ids)
        prompt = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(f"Your prompt is translated: {prompt}")
        return prompt
