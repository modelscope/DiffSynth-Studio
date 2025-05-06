from diffsynth import ModelManager, FluxImagePipeline, load_state_dict
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
from diffsynth.models.lora import FluxLoRAConverter
import torch, os, argparse
from diffsynth.pipelines.flux_image import lets_dance_flux
from accelerate import Accelerator
from tqdm import tqdm
import torch, os, json, torchvision
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoConfig, AutoTokenizer
import torch
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from diffsynth import ModelManager, FluxImagePipeline, load_state_dict, hash_state_dict_keys
from qwen_vl_utils import smart_resize
from PIL import Image
import numpy as np
import lightning as pl
os.environ["TOKENIZERS_PARALLELISM"] = "True"



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
        if skip_process:
            return image
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = self.image_process(image)
        return image
                
                
    def load_data(self, data_id):
        image_1, image_2, instruction = self.metadata[data_id]
        image_1 = self.load_image(image_1, skip_process=True) if image_1 is not None else None
        image_2 = self.load_image(image_2)
        return {"image_1": image_1, "image_2": image_2, "instruction": instruction}
        
        
    def __getitem__(self, data_id):
        if self.random:
            while True:
                try:
                    data_id = (torch.randint(0, len(self.metadata), (1,))[0] + data_id) % len(self.metadata)
                    data = self.load_data(data_id)
                    return data
                except:
                    continue
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



class UnifiedModel(pl.LightningModule):
    def __init__(self, flux_text_encoder_path, flux_vae_path, flux_dit_path, flux_decoder_path, qwenvl_path):
        super().__init__()
        # Load models
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([
            flux_text_encoder_path,
            flux_vae_path,
            flux_dit_path
        ])
        self.pipe = FluxImagePipeline.from_model_manager(model_manager)

        state_dict = load_state_dict(flux_decoder_path, torch_dtype=torch.bfloat16)
        self.pipe.dit.load_state_dict(state_dict, strict=False)

        self.adapter = torch.nn.Sequential(torch.nn.Linear(3584, 4096), torch.nn.LayerNorm(4096), torch.nn.ReLU(), torch.nn.Linear(4096, 4096), torch.nn.LayerNorm(4096)).to(dtype=torch.bfloat16)
        self.adapter.load_state_dict(state_dict, strict=False)

        self.qwenvl = NexusGenQwenVLEncoder.from_pretrained(qwenvl_path)
        
        self.pipe.vae_decoder.requires_grad_(False)
        self.pipe.vae_encoder.requires_grad_(False)
        self.pipe.text_encoder_1.requires_grad_(False)
        self.qwenvl.requires_grad_(False)
        
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        
    def training_step(self, batch, batch_idx):
        # Data
        text, image = batch["instruction"], batch["image_2"]
        image_ref = batch["image_1"]
        image = image.unsqueeze(0)

        # Prepare input parameters
        self.pipe.device = self.device
        latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        if image_ref is None:
            instruction = f"Generate an image according to the following description: {text}"
            images_ref = None
        else:
            instruction = f"<|vision_start|><|image_pad|><|vision_end|> {text}"
            images_ref = [image_ref]
        emb = self.qwenvl(instruction, images=images_ref)
        emb = self.adapter(emb)
        prompt_emb = self.pipe.encode_prompt("", positive=True, image_emb=emb)
        
        noise_pred = lets_dance_flux(
            self.pipe.denoising_model(),
            hidden_states=noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
            image_emb=emb,
            use_gradient_checkpointing=True
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        return loss


    def forward(self, batch):
        return self.training_step(batch, 0)
    
    
    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer



def search_for_last_checkpoint(path):
    if not os.path.exists(path):
        return None, 0
    checkpoint_list = os.listdir(path)
    checkpoint_list = [int(checkpoint.split("-")[1]) for checkpoint in checkpoint_list if checkpoint.startswith("epoch")]
    if len(checkpoint_list) == 0:
        return None, 0
    else:
        max_epoch_id = max(checkpoint_list)
        return f"{path}/epoch-{max_epoch_id}/model.safetensors", max_epoch_id + 1
    
    

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient_accumulation_steps",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="steps_per_epoch",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./models",
        help="output_path",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="learning_rate",
    )
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    model = UnifiedModel(
        "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
        "models/FLUX/FLUX.1-dev/ae.safetensors",
        "models/FLUX/FLUX.1-dev/flux1-dev.safetensors",
        "models/DiffSynth-Studio/Nexus-Gen/decoder_81_512.bin",
        "models/DiffSynth-Studio/Nexus-Gen",
    )
    # dataset and data loader
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    dataset = MultiTaskDataset(
        dataset_list=[
            SingleTaskDataset(
                "data/example_dataset",
                keys=(("image_1", "image_4", "editing_instruction"), ("image_4", "image_1", "reverse_editing_instruction"), (None, "image_1", "prompt")),
                height=512, width=512,
            ),
        ],
        dataset_weight=(1,),
        steps_per_epoch=args.steps_per_epoch * accelerator.num_processes,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=1,
        collate_fn=lambda x: x[0]
    )
    # train
    pretrained_path, start_epoch_id = search_for_last_checkpoint(args.output_path)
    if pretrained_path is not None:
        print(f"pretrained_path: {pretrained_path}")
        model.load_state_dict(load_state_dict(pretrained_path, torch_dtype=torch.bfloat16), assign=True, strict=False)

    model.to(torch.bfloat16)
    model.to(accelerator.device)
    
    trainable_modules = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_modules, lr=args.learning_rate)
    
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for epoch in range(start_epoch_id, 100000):
        for batch in tqdm(train_loader, desc=f"epoch-{epoch}", disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(batch)
                accelerator.backward(loss)
                optimizer.step()
        path = args.output_path
        os.makedirs(path, exist_ok=True)
        accelerator.wait_for_everyone()
        accelerator.save_model(model, f"{path}/epoch-{epoch}", max_shard_size="10GB", safe_serialization=True)
