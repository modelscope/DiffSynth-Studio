from .base_prompter import BasePrompter
from ..models.sd3_text_encoder import SD3TextEncoder1
from transformers import CLIPTokenizer, LlamaTokenizerFast, LlamaModel
import os, torch

PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>")

PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>")

PROMPT_TEMPLATE = {
    "dit-llm-encode": {
        "template": PROMPT_TEMPLATE_ENCODE,
        "crop_start": 36,
    },
    "dit-llm-encode-video": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO,
        "crop_start": 95,
    },
}

NEGATIVE_PROMPT = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"


class HunyuanVideoPrompter(BasePrompter):

    def __init__(
        self,
        tokenizer_1_path=None,
        tokenizer_2_path=None,
    ):
        if tokenizer_1_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_1_path = os.path.join(
                base_path, "tokenizer_configs/hunyuan_video/tokenizer_1")
        if tokenizer_2_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_2_path = os.path.join(
                base_path, "tokenizer_configs/hunyuan_video/tokenizer_2")
        super().__init__()
        self.tokenizer_1 = CLIPTokenizer.from_pretrained(tokenizer_1_path)
        self.tokenizer_2 = LlamaTokenizerFast.from_pretrained(tokenizer_2_path, padding_side='right')
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: LlamaModel = None

        self.prompt_template = PROMPT_TEMPLATE['dit-llm-encode']
        self.prompt_template_video = PROMPT_TEMPLATE['dit-llm-encode-video']

    def fetch_models(self, text_encoder_1: SD3TextEncoder1 = None, text_encoder_2: LlamaModel = None):
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2

    def apply_text_to_template(self, text, template):
        assert isinstance(template, str)
        if isinstance(text, list):
            return [self.apply_text_to_template(text_) for text_ in text]
        elif isinstance(text, str):
            # Will send string to tokenizer. Used for llm
            return template.format(text)
        else:
            raise TypeError(f"Unsupported prompt type: {type(text)}")

    def encode_prompt_using_clip(self, prompt, max_length, device):
        tokenized_result = self.tokenizer_1(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True
        )
        input_ids = tokenized_result.input_ids.to(device)
        attention_mask = tokenized_result.attention_mask.to(device)
        return self.text_encoder_1(input_ids=input_ids, extra_mask=attention_mask)[0]

    def encode_prompt_using_llm(self,
                                prompt,
                                max_length,
                                device,
                                crop_start,
                                hidden_state_skip_layer=2,
                                apply_final_norm=False,
                                use_attention_mask=True):
        max_length += crop_start
        inputs = self.tokenizer_2(prompt,
                                  return_tensors="pt",
                                  padding="max_length",
                                  max_length=max_length,
                                  truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        output_hidden_states = hidden_state_skip_layer is not None
        outputs = self.text_encoder_2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states)

        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            if hidden_state_skip_layer > 0 and apply_final_norm:
                last_hidden_state = self.text_encoder_2.norm(last_hidden_state)
        else:
            last_hidden_state = outputs['last_hidden_state']
        # crop out
        if crop_start > 0:
            last_hidden_state = last_hidden_state[:, crop_start:]
            attention_mask = (attention_mask[:, crop_start:] if use_attention_mask else None)

        return last_hidden_state, attention_mask

    def encode_prompt(self,
                      prompt,
                      positive=True,
                      device="cuda",
                      clip_sequence_length=77,
                      llm_sequence_length=256,
                      data_type='video',
                      use_template=True,
                      hidden_state_skip_layer=2,
                      apply_final_norm=False,
                      use_attention_mask=True):

        prompt = self.process_prompt(prompt, positive=positive)

        # apply template
        if use_template:
            template = self.prompt_template_video if data_type == 'video' else self.prompt_template
            prompt_formated = self.apply_text_to_template(prompt, template['template'])
        else:
            prompt_formated = prompt
        # Text encoder
        if data_type == 'video':
            crop_start = self.prompt_template_video.get("crop_start", 0)
        else:
            crop_start = self.prompt_template.get("crop_start", 0)

        # CLIP
        pooled_prompt_emb = self.encode_prompt_using_clip(prompt, clip_sequence_length, device)

        # LLM
        prompt_emb, attention_mask = self.encode_prompt_using_llm(
            prompt_formated, llm_sequence_length, device, crop_start,
            hidden_state_skip_layer, apply_final_norm, use_attention_mask)

        return prompt_emb, pooled_prompt_emb, attention_mask
