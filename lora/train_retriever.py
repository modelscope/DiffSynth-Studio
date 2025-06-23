from diffsynth import FluxImagePipeline, ModelManager, load_state_dict
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.pipelines.flux_image import lets_dance_flux
from lora.dataset import LoraDataset
from lora.retriever import TextEncoder, LoRAEncoder
from lora.utils import load_lora
import torch, os
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel



class LoRARetrieverTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.text_encoder = TextEncoder().to(torch.bfloat16)
        state_dict = load_state_dict("models/FLUX/FLUX.1-dev/text_encoder/model.safetensors")
        self.text_encoder.load_state_dict(TextEncoder.state_dict_converter().from_civitai(state_dict))
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        self.lora_encoder = LoRAEncoder().to(torch.bfloat16)
        
        self.tokenizer = CLIPTokenizer.from_pretrained("diffsynth/tokenizer_configs/stable_diffusion_3/tokenizer_1")
        
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self


    def forward(self, batch):
        text = [data["text"] for data in batch]
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).input_ids.to(self.device)
        text_emb = self.text_encoder(input_ids)
        text_emb = text_emb / text_emb.norm()
        
        lora_emb = []
        for data in batch:
            lora = FluxLoRAConverter.align_to_diffsynth_format(load_lora(data["model_file"], device=self.device))
            lora_emb.append(self.lora_encoder(lora))
        lora_emb = torch.concat(lora_emb)
        lora_emb = lora_emb / lora_emb.norm()
        
        similarity = text_emb @ lora_emb.T
        print(similarity)
        loss = -torch.log(torch.softmax(similarity, dim=0).diag()) - torch.log(torch.softmax(similarity, dim=1).diag())
        loss = 10 * loss.mean()
        return loss
    
    
    def trainable_modules(self):
        return self.lora_encoder.parameters()


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        
    
    def on_step_end(self, loss):
        pass
    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).lora_encoder.state_dict()
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


if __name__ == '__main__':
    model = LoRARetrieverTrainingModel()
    dataset = LoraDataset("data/lora/models/", "data/lora/lora_dataset_1000.csv", steps_per_epoch=100, loras_per_item=32)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1, collate_fn=lambda x: x[0])
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=1e-4)
    model_logger = ModelLogger("models/lora_retriever")
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    for epoch_id in range(1000000):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                print(loss)
        model_logger.on_epoch_end(accelerator, model, epoch_id)
