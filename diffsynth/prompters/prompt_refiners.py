from transformers import AutoTokenizer
from ..models.model_manager import ModelManager
import torch


class QwenPrompt(torch.nn.Modile):
    # This class leverages the open-source Qwen model to translate Chinese prompts into English, 
    #    with an integrated optimization mechanism for enhanced translation quality.
    def __init__(self, tokenizer_path=None, model=None, template=""):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.template = template

    @staticmethod
    def from_model_manager(model_nameger: ModelManager):
        model, model_path = model_nameger.fetch_model("qwen_prompt", require_model_path=True)

        template = 'Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {raw_prompt}\nOutput:'
        if model_path.endswith("v2"):
            template = """Converts a simple image description into a prompt. \
Prompts are formatted as multiple related tags separated by commas, plus you can use () to increase the weight, [] to decrease the weight, \
or use a number to specify the weight. You should add appropriate words to make the images described in the prompt more aesthetically pleasing, \
but make sure there is a correlation between the input and output.\n\
### Input: {raw_prompt}\n### Output:"""
        beautiful_prompt = QwenPrompt(
            tokenizer_path=model_path,
            model=model,
            template=template
        )
        return qwen_prompt

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
            print(f"Your prompt is refined by Qwen : {prompt}")
            return prompt
        else:
            return raw_prompt


class BeautifulPrompt(torch.nn.Module):
    def __init__(self, tokenizer_path=None, model=None, template=""):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.template = template


    @staticmethod
    def from_model_manager(model_nameger: ModelManager):
        model, model_path = model_nameger.fetch_model("beautiful_prompt", require_model_path=True)
        system_prompt = """你是一个英文图片描述家，你看到一段中文图片描述后，尽可能用精简准确的英文，将中文的图片描述的意境用英文短句展示出来，并附带图片风格描述，如果中文描述中没有明确的风格，你需要根据中文意境额外添加一些风格描述，确保图片中的内容丰富生动。\n\n你有如下几种不同的风格描述示例进行参考：\n\n特写风格: Extreme close-up by Oliver Dum, magnified view of a dewdrop on a spider web occupying the frame, the camera focuses closely on the object with the background blurred. The image is lit with natural sunlight, enhancing the vivid textures and contrasting colors.\n\n复古风格: Photograph of women working, Daguerreotype, calotype, tintype, collodion, ambrotype, carte-de-visite, gelatin silver, dry plate, wet plate, stereoscope, albumen print, cyanotype, glass, lantern slide, camera  \n\n动漫风格： a happy dairy cow just finished grazing, in the style of cartoon realism, disney animation, hyper-realistic portraits, 32k uhd, cute cartoonish designs, wallpaper, luminous brushwork  \n\n普通人物场景风格： A candid shot of young best friends dirty, at the skatepark, natural afternoon light, Canon EOS R5, 100mm, F 1.2 aperture setting capturing a moment, cinematic \n\n景观风格: bright beautiful sunrise over the sea and rocky mountains, photorealistic, \n\n设计风格: lionface circle tshirt design, in the style of detailed botanical illustrations, colorful cartoon, exotic atmosphere, 2d game art, white background, contour \n\n动漫风格: Futuristic mecha robot walking through a neon cityscape, with lens flares, dramatic lighting, illustrated like a Gundam anime poster \n\n都市风格: warmly lit room with large monitors on the clean desk, overlooking the city, ultrareal and photorealistic,  \n\n\n请根据上述图片风格，以及中文描述生成对应的英文图片描述  \n\n 请注意：\n\n 如果中文为成语或古诗，不能只根据表层含义来进行描述，而要描述其中的意境！例如：“胸有成竹”的图片场景中并没有竹子，而是描述一个人非常自信的场景，请在英文翻译中不要提到bamboo，以此类推\n\n字数不超过100字"""
        messages = [{
            'role': 'system',
            'content': system_prompt
        }, {
            'role': 'user',
            'content': "{raw_prompt}"
        }]

        qwen_prompt = QwenPrompt(
            tokenizer_path=model_path,
            model=model,
            template=template
        )
        return qwen_prompt
    

    def __call__(self, raw_prompt, positive=True, **kwargs):
        if positive:
            model_input = self.template.format(raw_prompt=raw_prompt)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            prompt = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"Your prompt is refined by Qwen: {prompt}, RawPrompt: {raw_prompt}")
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
