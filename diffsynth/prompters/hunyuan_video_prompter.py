from .base_prompter import BasePrompter
from ..models.sd3_text_encoder import SD3TextEncoder1
from ..models.hunyuan_video_text_encoder import HunyuanVideoLLMEncoder, HunyuanVideoMLLMEncoder
from transformers import CLIPTokenizer, LlamaTokenizerFast, CLIPImageProcessor
import os, torch
from typing import Union

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

PROMPT_TEMPLATE_ENCODE_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

PROMPT_TEMPLATE = {
    "dit-llm-encode": {
        "template": PROMPT_TEMPLATE_ENCODE,
        "crop_start": 36,
    },
    "dit-llm-encode-video": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO,
        "crop_start": 95,
    },
    "dit-llm-encode-i2v": {
        "template": PROMPT_TEMPLATE_ENCODE_I2V,
        "crop_start": 36,
        "image_emb_start": 5,
        "image_emb_end": 581,
        "image_emb_len": 576,
        "double_return_token_id": 271
    },
    "dit-llm-encode-video-i2v": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO_I2V,
        "crop_start": 103,
        "image_emb_start": 5,
        "image_emb_end": 581,
        "image_emb_len": 576,
        "double_return_token_id": 271
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
        self.text_encoder_2: HunyuanVideoLLMEncoder = None

        self.prompt_template = PROMPT_TEMPLATE['dit-llm-encode']
        self.prompt_template_video = PROMPT_TEMPLATE['dit-llm-encode-video']

    def fetch_models(self,
                     text_encoder_1: SD3TextEncoder1 = None,
                     text_encoder_2: Union[HunyuanVideoLLMEncoder, HunyuanVideoMLLMEncoder] = None):
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        if isinstance(text_encoder_2, HunyuanVideoMLLMEncoder):
            # processor
            # TODO: may need to replace processor with local implementation
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_2_path = os.path.join(base_path, "tokenizer_configs/hunyuan_video/tokenizer_2")
            self.processor = CLIPImageProcessor.from_pretrained(tokenizer_2_path)
            # template
            self.prompt_template = PROMPT_TEMPLATE['dit-llm-encode-i2v']
            self.prompt_template_video = PROMPT_TEMPLATE['dit-llm-encode-video-i2v']

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
                                use_attention_mask=True):
        max_length += crop_start
        inputs = self.tokenizer_2(prompt,
                                  return_tensors="pt",
                                  padding="max_length",
                                  max_length=max_length,
                                  truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        last_hidden_state = self.text_encoder_2(input_ids, attention_mask, hidden_state_skip_layer)

        # crop out
        if crop_start > 0:
            last_hidden_state = last_hidden_state[:, crop_start:]
            attention_mask = (attention_mask[:, crop_start:] if use_attention_mask else None)

        return last_hidden_state, attention_mask

    def encode_prompt_using_mllm(self,
                                prompt,
                                images,
                                max_length,
                                device,
                                crop_start,
                                hidden_state_skip_layer=2,
                                use_attention_mask=True,
                                image_embed_interleave=4):
        image_outputs = self.processor(images, return_tensors="pt")["pixel_values"].to(device)
        max_length += crop_start
        inputs = self.tokenizer_2(prompt,
                                  return_tensors="pt",
                                  padding="max_length",
                                  max_length=max_length,
                                  truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        last_hidden_state = self.text_encoder_2(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                hidden_state_skip_layer=hidden_state_skip_layer,
                                                pixel_values=image_outputs)

        text_crop_start = (crop_start - 1 + self.prompt_template_video.get("image_emb_len", 576))
        image_crop_start = self.prompt_template_video.get("image_emb_start", 5)
        image_crop_end = self.prompt_template_video.get("image_emb_end", 581)
        batch_indices, last_double_return_token_indices = torch.where(
            input_ids == self.prompt_template_video.get("double_return_token_id", 271))
        if last_double_return_token_indices.shape[0] == 3:
            # in case the prompt is too long
            last_double_return_token_indices = torch.cat((
                last_double_return_token_indices,
                torch.tensor([input_ids.shape[-1]]),
            ))
            batch_indices = torch.cat((batch_indices, torch.tensor([0])))
        last_double_return_token_indices = (last_double_return_token_indices.reshape(input_ids.shape[0], -1)[:, -1])
        batch_indices = batch_indices.reshape(input_ids.shape[0], -1)[:, -1]
        assistant_crop_start = (last_double_return_token_indices - 1 + self.prompt_template_video.get("image_emb_len", 576) - 4)
        assistant_crop_end = (last_double_return_token_indices - 1 + self.prompt_template_video.get("image_emb_len", 576))
        attention_mask_assistant_crop_start = (last_double_return_token_indices - 4)
        attention_mask_assistant_crop_end = last_double_return_token_indices
        text_last_hidden_state = []
        text_attention_mask = []
        image_last_hidden_state = []
        image_attention_mask = []
        for i in range(input_ids.shape[0]):
            text_last_hidden_state.append(
                torch.cat([
                    last_hidden_state[i, text_crop_start:assistant_crop_start[i].item()],
                    last_hidden_state[i, assistant_crop_end[i].item():],
                ]))
            text_attention_mask.append(
                torch.cat([
                    attention_mask[
                        i,
                        crop_start:attention_mask_assistant_crop_start[i].item(),
                    ],
                    attention_mask[i, attention_mask_assistant_crop_end[i].item():],
                ]) if use_attention_mask else None)
            image_last_hidden_state.append(last_hidden_state[i, image_crop_start:image_crop_end])
            image_attention_mask.append(
                torch.ones(image_last_hidden_state[-1].shape[0]).to(last_hidden_state.device).
                to(attention_mask.dtype) if use_attention_mask else None)

        text_last_hidden_state = torch.stack(text_last_hidden_state)
        text_attention_mask = torch.stack(text_attention_mask)
        image_last_hidden_state = torch.stack(image_last_hidden_state)
        image_attention_mask = torch.stack(image_attention_mask)

        image_last_hidden_state = image_last_hidden_state[:, ::image_embed_interleave, :]
        image_attention_mask = image_attention_mask[:, ::image_embed_interleave]

        assert (text_last_hidden_state.shape[0] == text_attention_mask.shape[0] and
                image_last_hidden_state.shape[0] == image_attention_mask.shape[0])

        last_hidden_state = torch.cat([image_last_hidden_state, text_last_hidden_state], dim=1)
        attention_mask = torch.cat([image_attention_mask, text_attention_mask], dim=1)

        return last_hidden_state, attention_mask

    def encode_prompt(self,
                      prompt,
                      images=None,
                      positive=True,
                      device="cuda",
                      clip_sequence_length=77,
                      llm_sequence_length=256,
                      data_type='video',
                      use_template=True,
                      hidden_state_skip_layer=2,
                      use_attention_mask=True,
                      image_embed_interleave=4):

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
        if images is None:
            prompt_emb, attention_mask = self.encode_prompt_using_llm(prompt_formated, llm_sequence_length, device, crop_start,
                                                                      hidden_state_skip_layer, use_attention_mask)
        else:
            prompt_emb, attention_mask = self.encode_prompt_using_mllm(prompt_formated, images, llm_sequence_length, device,
                                                                       crop_start, hidden_state_skip_layer, use_attention_mask,
                                                                       image_embed_interleave)

        return prompt_emb, pooled_prompt_emb, attention_mask
