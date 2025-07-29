import torch
from PIL import Image
from qwen_vl_utils import smart_resize
from transformers import AutoConfig
from .nexus_gen_ar_model import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


class NexusGenAutoregressiveModel(torch.nn.Module):
    def __init__(self, model_path="models/DiffSynth-Studio/Nexus-GenV2", max_length=1024, max_pixels=262640, dtype=torch.bfloat16, device="cuda"):
        super(NexusGenAutoregressiveModel, self).__init__()
        self.max_length = max_length
        self.max_pixels = max_pixels
        model_config = AutoConfig.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration(model_config)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)


    @staticmethod
    def state_dict_converter():
        return NexusGenAutoregressiveModelStateDictConverter()

    def bound_image(self, image, max_pixels=262640):
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            max_pixels=max_pixels,
        )
        return image.resize((resized_width, resized_height))

    def get_editing_msg(self, instruction):
        if '<image>' not in instruction:
            instruction = '<image> ' + instruction
        messages = [{"role":"user", "content":instruction}, {"role":"assistant", "content":"Here is the image: <image>"}]
        return messages

    def get_generation_msg(self, instruction):
        instruction = "Generate an image according to the following description: {}".format(instruction)
        messages = [{"role":"user", "content":instruction}, {"role":"assistant", "content":"Here is an image based on the description: <image>"}]
        return messages

    def forward(self, instruction, ref_image=None, num_img_tokens=81):
        """
        Generate target embeddings for the given instruction and reference image.
        """
        if ref_image is not None:
            messages = self.get_editing_msg(instruction)
            images = [self.bound_image(ref_image)] + [Image.new(mode='RGB', size=(252, 252), color=(255, 255, 255))]
            output_image_embeddings = self.get_target_embeddings(images, messages, self.processor, self.model, num_img_tokens)
        else:
            messages = self.get_generation_msg(instruction)
            images = [Image.new(mode='RGB', size=(252, 252), color=(255, 255, 255))]
            output_image_embeddings = self.get_target_embeddings(images, messages, self.processor, self.model, num_img_tokens)

        return output_image_embeddings

    def get_target_embeddings(self, images, messages, processor, model, num_img_tokens=81):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        text = text.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        inputs = processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        input_embeds = model.model.embed_tokens(inputs['input_ids'])
        image_embeds = model.visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
        ground_truth_image_embeds = image_embeds[-num_img_tokens:]
        input_image_embeds = image_embeds[:-num_img_tokens]

        image_mask = inputs['input_ids'] == model.config.image_token_id
        indices = image_mask.cumsum(dim=1)
        input_image_mask = torch.logical_and(indices <= (image_embeds.shape[0] - ground_truth_image_embeds.shape[0]), image_mask)
        gt_image_mask = torch.logical_and(image_mask, ~input_image_mask)
        input_image_mask = input_image_mask.unsqueeze(-1).expand_as(input_embeds)
        input_embeds = input_embeds.masked_scatter(input_image_mask, input_image_embeds)

        image_prefill_embeds = model.image_prefill_embeds(
            torch.arange(81, device=model.device).long()
        )
        input_embeds = input_embeds.masked_scatter(gt_image_mask.unsqueeze(-1).expand_as(input_embeds), image_prefill_embeds)

        position_ids, _ = model.get_rope_index(
            inputs['input_ids'],
            inputs['image_grid_thw'],
            attention_mask=inputs['attention_mask'])
        position_ids = position_ids.contiguous()
        outputs = model(inputs_embeds=input_embeds, position_ids=position_ids, attention_mask=inputs['attention_mask'], return_dict=True)
        output_image_embeddings = outputs.image_embeddings[:, :-1, :]
        output_image_embeddings = output_image_embeddings[gt_image_mask[:, 1:]]
        return output_image_embeddings, input_image_embeds, inputs['image_grid_thw']


class NexusGenAutoregressiveModelStateDictConverter:
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        state_dict = {"model." + key: value for key, value in state_dict.items()}
        return state_dict
