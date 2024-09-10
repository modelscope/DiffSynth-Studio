from transformers import AutoTokenizer
from ..models.model_manager import ModelManager
import torch
from .omost import OmostPromter

class BeautifulPrompt(torch.nn.Module):
    def __init__(self, tokenizer_path=None, model=None, template=""):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.template = template


    @staticmethod
    def from_model_manager(model_manager: ModelManager):
        model, model_path = model_manager.fetch_model("beautiful_prompt", require_model_path=True)
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



class QwenPrompt(torch.nn.Module):
    # This class leverages the open-source Qwen model to translate Chinese prompts into English, 
    #    with an integrated optimization mechanism for enhanced translation quality.
    def __init__(self, tokenizer_path=None, model=None, system_prompt=""):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.system_prompt = system_prompt


    @staticmethod
    def from_model_manager(model_nameger: ModelManager):
        model, model_path = model_nameger.fetch_model("qwen_prompt", require_model_path=True)
        system_prompt = """You are an English image describer. Here are some example image styles:\n\n1. Extreme close-up: Clear focus on a single object with a blurred background, highlighted under natural sunlight.\n2. Vintage: A photograph of a historical scene, using techniques such as Daguerreotype or cyanotype.\n3. Anime: A stylized cartoon image, emphasizing hyper-realistic portraits and luminous brushwork.\n4. Candid: A natural, unposed shot capturing spontaneous moments, often with cinematic qualities.\n5. Landscape: A photorealistic image of natural scenery, such as a sunrise over the sea.\n6. Design: Colorful and detailed illustrations, often in the style of 2D game art or botanical illustrations.\n7. Urban: An ultrarealistic scene in a modern setting, possibly a cityscape viewed from indoors.\n\nYour task is to translate a given Chinese image description into a concise and precise English description. Ensure that the imagery is vivid and descriptive, and include stylistic elements to enrich the description.\nPlease note the following points:\n\n1. Capture the essence and mood of the Chinese description without including direct phrases or words from the examples provided.\n2. You should add appropriate words to make the images described in the prompt more aesthetically pleasing. If the Chinese description does not specify a style, you need to add some stylistic descriptions based on the essence of the Chinese text.\n3. The generated English description should not exceed 200 words.\n\n"""
        qwen_prompt = QwenPrompt(
            tokenizer_path=model_path,
            model=model,
            system_prompt=system_prompt
        )
        return qwen_prompt


    def __call__(self, raw_prompt, positive=True, **kwargs):
        if positive:
            messages = [{
                'role': 'system',
                'content': self.system_prompt
            }, {
                'role': 'user',
                'content': raw_prompt
            }]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            prompt = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"Your prompt is refined by Qwen: {prompt}")
            return prompt
        else:
            return raw_prompt



class Translator(torch.nn.Module):
    def __init__(self, tokenizer_path=None, model=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model


    @staticmethod
    def from_model_manager(model_manager: ModelManager):
        model, model_path = model_manager.fetch_model("translator", require_model_path=True)
        translator = Translator(tokenizer_path=model_path, model=model)
        return translator
    

    def __call__(self, prompt, **kwargs):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        output_ids = self.model.generate(input_ids)
        prompt = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(f"Your prompt is translated: {prompt}")
        return prompt
