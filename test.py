from transformers import AutoConfig, AutoTokenizer
import torch
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from diffsynth import ModelManager, FluxImagePipeline, load_state_dict
from qwen_vl_utils import smart_resize
from PIL import Image



class NexusGenQwenVLEncoder(torch.nn.Module):
    def __init__(self, model_path, torch_dtype="auto", device="cpu"):
        super().__init__()
        model_config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=model_config, trust_remote_code=True, torch_dtype=torch_dtype, device_map=device)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    
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

    def forward(self, prompt, images=None):
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": prompt
            },],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=self.process_images(images),
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generation_image_grid_thw = torch.tensor([[1, 18, 18]]).to(self.model.device)
        outputs = self.model.generate(**inputs,
                                        max_new_tokens=1024,
                                        return_dict_in_generate=True,
                                        generation_image_grid_thw=generation_image_grid_thw)
        output_image_embeddings = outputs['output_image_embeddings']
        return output_image_embeddings





model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
pipe = FluxImagePipeline.from_model_manager(model_manager)
pipe.dit.load_state_dict(load_state_dict("models/DiffSynth-Studio/Nexus-Gen/decoder_81_512.bin", torch_dtype=torch.bfloat16), strict=False)

qwenvl = NexusGenQwenVLEncoder.from_pretrained('models/DiffSynth-Studio/Nexus-Gen').to("cuda")

adapter = torch.nn.Sequential(torch.nn.Linear(3584, 4096), torch.nn.LayerNorm(4096), torch.nn.ReLU(), torch.nn.Linear(4096, 4096), torch.nn.LayerNorm(4096)).to(dtype=torch.bfloat16, device="cuda")
adapter.load_state_dict(load_state_dict("models/DiffSynth-Studio/Nexus-Gen/decoder_81_512.bin", torch_dtype=torch.bfloat16), strict=False)

with torch.no_grad():
    instruction = "<|vision_start|><|image_pad|><|vision_end|> Transform the style to flat anime. Keep the color."
    emb = qwenvl(instruction, images=[Image.open("image_3.jpg").convert('RGB')])
    emb = adapter(emb)
    image = pipe("", image_emb=emb)
    image.save("image_4.jpg")
