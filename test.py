from transformers import AutoConfig, AutoTokenizer
import torch
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from diffsynth import ModelManager, FluxImagePipeline, load_state_dict, hash_state_dict_keys
from qwen_vl_utils import smart_resize
from PIL import Image
import numpy as np



class NexusGenQwenVLEncoder(torch.nn.Module):
    def __init__(self, model_path, torch_dtype="auto", device="cpu"):
        super().__init__()
        model_config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=model_config, trust_remote_code=True, torch_dtype=torch_dtype, device_map=device)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.t2i_template = "Here is an image based on the description: <|vision_start|><|image_pad|><|vision_end|>"
        self.i2i_template = "Here is the image: <|vision_start|><|image_pad|><|vision_end|>"
    
    @staticmethod
    def from_pretrained(model_path, torch_dtype="auto", device="cpu"):
        return NexusGenQwenVLEncoder(model_path, torch_dtype=torch_dtype, device=device).eval()
    
    def process_images(self, images=None):
        if images is None:
            return None
        # resize input to max_pixels to avoid oom
        for j in range(len(images)):
            input_image = images[j]
            input_w, input_h = input_image.size
            resized_height, resized_width = smart_resize(
                input_h,
                input_w,
                max_pixels=262640,
            )
            images[j] = input_image.resize((resized_width, resized_height))
        return images

    def forward(self, prompt, images=None, num_img_tokens=81):
        messages = [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                },],
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": self.t2i_template if images is None else self.i2i_template
                },],
            }
        ]
        images = self.process_images(images)
        target_image = Image.fromarray(np.zeros((252, 252, 3), dtype=np.uint8))
        if images is None:
            images = [target_image]
        else:
            images = images + [target_image]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        input_embeds = self.model.model.embed_tokens(inputs['input_ids'])
        image_embeds = self.model.visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
        ground_truth_image_embeds = image_embeds[-num_img_tokens:]
        input_image_embeds = image_embeds[:-num_img_tokens]

        image_mask = inputs['input_ids'] == self.model.config.image_token_id
        indices = image_mask.cumsum(dim=1)
        input_image_mask = torch.logical_and(indices <= (image_embeds.shape[0] - ground_truth_image_embeds.shape[0]), image_mask)
        gt_image_mask = torch.logical_and(image_mask, ~input_image_mask)
        input_image_mask = input_image_mask.unsqueeze(-1).expand_as(input_embeds)
        input_embeds = input_embeds.masked_scatter(input_image_mask, input_image_embeds)

        position_ids, _ = self.model.get_rope_index(inputs['input_ids'],
                                                    inputs['image_grid_thw'],
                                                    attention_mask=inputs['attention_mask'])
        position_ids = position_ids.contiguous()
        outputs = self.model(inputs_embeds=input_embeds, position_ids=position_ids, attention_mask=inputs['attention_mask'], return_dict=True)
        output_image_embeddings = outputs.image_embeddings[:, :-1, :] # shift right
        output_image_embeddings = output_image_embeddings[gt_image_mask[:, 1:]]
        output_image_embeddings = output_image_embeddings.unsqueeze(0)
        return output_image_embeddings



model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
pipe = FluxImagePipeline.from_model_manager(model_manager)

state_dict = load_state_dict("models/DiffSynth-Studio/Nexus-Gen/decoder_81_512.bin", torch_dtype=torch.bfloat16)
pipe.dit.load_state_dict(state_dict, strict=False)

adapter = torch.nn.Sequential(torch.nn.Linear(3584, 4096), torch.nn.LayerNorm(4096), torch.nn.ReLU(), torch.nn.Linear(4096, 4096), torch.nn.LayerNorm(4096)).to(dtype=torch.bfloat16, device="cuda")
adapter.load_state_dict(state_dict, strict=False)

qwenvl = NexusGenQwenVLEncoder.from_pretrained('models/DiffSynth-Studio/Nexus-Gen').to("cuda")

with torch.no_grad():
    instruction = "Generate an image according to the following description: a beautiful Asian girl"
    emb = qwenvl(instruction, images=None)
    emb = adapter(emb)
    image = pipe("", image_emb=emb)
    image.save("image_1.jpg")
    
with torch.no_grad():
    instruction = "<|vision_start|><|image_pad|><|vision_end|> Add sunglasses."
    emb = qwenvl(instruction, images=[Image.open("image_1.jpg")])
    emb = adapter(emb)
    image = pipe("", image_emb=emb)
    image.save("image_2.jpg")

with torch.no_grad():
    instruction = "<|vision_start|><|image_pad|><|vision_end|> Let her smile."
    emb = qwenvl(instruction, images=[Image.open("image_2.jpg")])
    emb = adapter(emb)
    image = pipe("", image_emb=emb)
    image.save("image_3.jpg")
