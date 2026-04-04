import torch
from PIL import Image


class NexusGenAutoregressiveModel(torch.nn.Module):
    def __init__(self, max_length=1024, max_pixels=262640):
        super(NexusGenAutoregressiveModel, self).__init__()
        from .nexus_gen_ar_model import Qwen2_5_VLForConditionalGeneration
        from transformers import Qwen2_5_VLConfig
        self.max_length = max_length
        self.max_pixels = max_pixels
        model_config = Qwen2_5_VLConfig(**{
            "_name_or_path": "DiffSynth-Studio/Nexus-GenV2",
            "architectures": [
                "Qwen2_5_VLForConditionalGeneration"
            ],
            "attention_dropout": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_qwen2_5_vl.Qwen2_5_VLConfig",
                "AutoModel": "modeling_qwen2_5_vl.Qwen2_5_VLModel",
                "AutoModelForCausalLM": "modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration"
            },
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 3584,
            "image_token_id": 151655,
            "initializer_range": 0.02,
            "intermediate_size": 18944,
            "max_position_embeddings": 128000,
            "max_window_layers": 28,
            "model_type": "qwen2_5_vl",
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "pad_token_id": 151643,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "mrope_section": [
                16,
                24,
                24
                ],
                "rope_type": "default",
                "type": "default"
            },
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.49.0",
            "use_cache": False,
            "use_sliding_window": False,
            "video_token_id": 151656,
            "vision_config": {
                "hidden_size": 1280,
                "in_chans": 3,
                "model_type": "qwen2_5_vl",
                "spatial_patch_size": 14,
                "tokens_per_second": 2,
                "torch_dtype": "bfloat16"
            },
            "vision_end_token_id": 151653,
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "vocab_size": 152064
        })
        self.model = Qwen2_5_VLForConditionalGeneration(model_config)
        self.processor = None
        
        
    def load_processor(self, path):
        from .nexus_gen_ar_model import Qwen2_5_VLProcessor
        self.processor = Qwen2_5_VLProcessor.from_pretrained(path)


    @staticmethod
    def state_dict_converter():
        return NexusGenAutoregressiveModelStateDictConverter()

    def bound_image(self, image, max_pixels=262640):
        from qwen_vl_utils import smart_resize
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
