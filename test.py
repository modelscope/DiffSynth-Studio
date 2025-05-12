from transformers import AutoConfig, AutoTokenizer
import torch, json, os, torchvision
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from diffsynth import ModelManager, FluxImagePipeline, load_state_dict, hash_state_dict_keys
from qwen_vl_utils import smart_resize
from PIL import Image
import numpy as np
from torchvision.transforms import v2



class SingleTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        keys=(("image_1", "image_2", "editing_instruction"), ("image_2", "image_1", "reverse_editing_instruction"), (None, "image_1", "prompt")),
        height=1024, width=1024, random=True, steps_per_epoch=1000, metadata_path=None
    ):
        self.base_path = base_path
        self.keys = keys
        self.metadata = []
        self.bad_data = []
        self.height = height
        self.width = width
        self.random = random
        self.steps_per_epoch = steps_per_epoch
        self.image_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        if metadata_path is None:
            self.search_for_data("", report_data_log=True)
            self.report_data_log()
        else:
            with open(metadata_path, "r", encoding="utf-8-sig") as f:
                self.metadata = json.load(f)


    def report_data_log(self):
        print(f"{len(self.metadata)} valid data, {len(self.bad_data)} invalid data.")


    def dump_metadata(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)
        
        
    def parse_json_file(self, absolute_path, relative_path):
        data_list = []
        with open(absolute_path, "r") as f:
            metadata = json.load(f)
            for image_1, image_2, instruction in self.keys:
                image_1 = os.path.join(relative_path, metadata[image_1]) if image_1 is not None else None
                image_2 = os.path.join(relative_path, metadata[image_2])
                instruction = metadata[instruction]
                data_list.append((image_1, image_2, instruction))
        return data_list
    
        
    def search_for_data(self, path, report_data_log=False):
        now_path = os.path.join(self.base_path, path)
        if os.path.isfile(now_path) and path.endswith(".json"):
            try:
                data_list = self.parse_json_file(now_path, os.path.dirname(path))
                self.metadata.extend(data_list)
            except:
                self.bad_data.append(now_path)
        elif os.path.isdir(now_path):
            for sub_path in os.listdir(now_path):
                self.search_for_data(os.path.join(path, sub_path))
                if report_data_log and os.path.isdir(os.path.join(self.base_path, path, sub_path)):
                    self.report_data_log()
                
                
    def load_image(self, image_path, skip_process=False):
        image_path = os.path.join(self.base_path, image_path)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        if skip_process:
            return image
        image = self.image_process(image)
        return image
                
                
    def load_data(self, data_id):
        image_1, image_2, instruction = self.metadata[data_id]
        image_1 = self.load_image(image_1, skip_process=True) if image_1 is not None else None
        image_2 = self.load_image(image_2)
        return {"image_1": image_1, "image_2": image_2, "instruction": instruction}
        
        
    def __getitem__(self, data_id):
        if self.random:
            data_id = (torch.randint(0, len(self.metadata), (1,))[0] + data_id) % len(self.metadata)
            data = self.load_data(data_id)
            return data
        else:
            return self.load_data(data_id)


    def __len__(self):
        return self.steps_per_epoch if self.random else len(self.metadata)
    
    
    
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list, dataset_weight, steps_per_epoch=1000):
        self.dataset_list = dataset_list
        self.dataset_weight = torch.tensor(dataset_weight, dtype=torch.float)
        self.steps_per_epoch = steps_per_epoch

        
    def __getitem__(self, data_id):
        dataset_id = torch.multinomial(self.dataset_weight, 1).tolist()[0]
        data_id = torch.randint(0, len(self.dataset_list[dataset_id]), (1,))[0]
        data = self.dataset_list[dataset_id][data_id]
        return data


    def __len__(self):
        return self.steps_per_epoch



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

# state_dict = load_state_dict("models/DiffSynth-Studio/Nexus-Gen/decoder_81_512.bin", torch_dtype=torch.bfloat16)
# pipe.dit.load_state_dict(state_dict, strict=False)

adapter = torch.nn.Sequential(torch.nn.Linear(3584, 4096), torch.nn.LayerNorm(4096), torch.nn.ReLU(), torch.nn.Linear(4096, 4096), torch.nn.LayerNorm(4096)).to(dtype=torch.bfloat16, device="cuda")
# adapter.load_state_dict(state_dict, strict=False)

qwenvl = NexusGenQwenVLEncoder.from_pretrained('models/DiffSynth-Studio/Nexus-Gen').to("cuda")

sd = {}
for i in range(1, 6):
    print(i)
    sd.update(load_state_dict(f"models/nexus_v3/epoch-19/model-0000{i}-of-00005.safetensors", torch_dtype=torch.bfloat16))
pipe.dit.load_state_dict({i.replace("pipe.dit.", ""): sd[i] for i in sd if i.startswith("pipe.dit.")})
qwenvl.load_state_dict({i.replace("qwenvl.", ""): sd[i] for i in sd if i.startswith("qwenvl.")})
adapter.load_state_dict({i.replace("adapter.", ""): sd[i] for i in sd if i.startswith("adapter.")})


dataset = MultiTaskDataset(
    dataset_list=[
        SingleTaskDataset(
            "/shark/zhongjie/data/image_pulse_datasets/task1/data/dataset_change_add_remove",
            keys=(("image_1", "image_2", "editing_instruction"), ("image_2", "image_1", "reverse_editing_instruction"), (None, "image_1", "prompt")),
            height=1024, width=1024,
            metadata_path="/shark/zhongjie/data/image_pulse_datasets/task1/data/metadata/20250507_dataset_change_add_remove.json",
        ),
        SingleTaskDataset(
            "/shark/zhongjie/data/image_pulse_datasets/task1/data/dataset_style_transfer",
            keys=(("image_1", "image_4", "editing_instruction"), ("image_4", "image_1", "reverse_editing_instruction"), (None, "image_1", "prompt")),
            height=1024, width=1024,
            metadata_path="/shark/zhongjie/data/image_pulse_datasets/task1/data/metadata/20250507_dataset_style_transfer.json",
        ),
        SingleTaskDataset(
            "/shark/zhongjie/data/image_pulse_datasets/task1/data/dataset_faceid",
            keys=(("image_1", "image_2", "editing_instruction"), ("image_2", "image_1", "reverse_editing_instruction")),
            height=1024, width=1024,
            metadata_path="/shark/zhongjie/data/image_pulse_datasets/task1/data/metadata/20250507_dataset_faceid.json",
        ),
    ],
    dataset_weight=(4, 2, 1,),
    steps_per_epoch=100000
)


torch.manual_seed(0)
for data_id, data in enumerate(dataset):
    image_1 = data["image_1"]
    image_2 = data["image_2"].cpu().float().permute(1, 2, 0).numpy()
    image_2 = Image.fromarray(((image_2 / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
    instruction = data["instruction"]

    print(instruction)
    if image_1 is None:
        with torch.no_grad():
            instruction = f"Generate an image according to the following description: {instruction}"
            emb = qwenvl(instruction, images=None)
            emb = adapter(emb)
            image_3 = pipe("", image_emb=emb)
    else:
        with torch.no_grad():
            instruction = f"<|vision_start|><|image_pad|><|vision_end|> {instruction}"
            emb = qwenvl(instruction, images=[image_1])
            emb = adapter(emb)
            image_3 = pipe("", image_emb=emb)
    
    if image_1 is not None:
        image_1.save(f"data/output/{data_id}_1.jpg")
    image_2.save(f"data/output/{data_id}_2.jpg")
    image_3.save(f"data/output/{data_id}_3.jpg")
    if data_id >= 100:
        break



# with torch.no_grad():
#     instruction = "Generate an image according to the following description: hyper-realistic and detailed 2010s movie still portrait of Josip Broz Tito, by Paolo Sorrentino, Leica SL2 50mm, clear color, high quality, high textured, dramatic light, cinematic"
#     emb = qwenvl(instruction, images=None)
#     emb = adapter(emb)
#     image = pipe("", image_emb=emb)
#     image.save("image_1.jpg")
    
# with torch.no_grad():
#     instruction = "<|vision_start|><|image_pad|><|vision_end|> transform the image into a cartoon style with vibrant colors and a confident expression."
#     emb = qwenvl(instruction, images=[Image.open("image_1.jpg")])
#     emb = adapter(emb)
#     image = pipe("", image_emb=emb)
#     image.save("image_2.jpg")
