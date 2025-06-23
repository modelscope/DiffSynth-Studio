from diffsynth import FluxImagePipeline, ModelManager, load_state_dict
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.pipelines.flux_image import lets_dance_flux
from lora.dataset import LoraDataset
from lora.retriever import TextEncoder, LoRAEncoder
from lora.merger import LoraPatcher
from lora.utils import load_lora
import torch, os
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel
import pandas as pd



class LoRARetrieverTrainingModel(torch.nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        
        self.text_encoder = TextEncoder().to(torch.bfloat16)
        state_dict = load_state_dict("models/FLUX/FLUX.1-dev/text_encoder/model.safetensors")
        self.text_encoder.load_state_dict(TextEncoder.state_dict_converter().from_civitai(state_dict))
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        self.lora_encoder = LoRAEncoder().to(torch.bfloat16)
        state_dict = load_state_dict(pretrained_path)
        self.lora_encoder.load_state_dict(state_dict)
        
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
    
    @torch.no_grad()
    def process_lora_list(self, lora_list):
        lora_emb = []
        for lora in tqdm(lora_list):
            lora = FluxLoRAConverter.align_to_diffsynth_format(load_lora(lora, device="cuda"))
            lora_emb.append(self.lora_encoder(lora))
        lora_emb = torch.concat(lora_emb)
        lora_emb = lora_emb / lora_emb.norm()
        self.lora_emb = lora_emb
        self.lora_list = lora_list
    
    @torch.no_grad()
    def retrieve(self, text, k=1):
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).input_ids.to(self.device)
        text_emb = self.text_encoder(input_ids)
        text_emb = text_emb / text_emb.norm()
        
        similarity = text_emb @ self.lora_emb.T
        topk = torch.topk(similarity, k, dim=1).indices[0]
        
        lora_list = []
        model_url_list = []
        for lora_id in topk:
            print(self.lora_list[lora_id])
            lora = FluxLoRAConverter.align_to_diffsynth_format(load_lora(self.lora_list[lora_id], device="cuda"))
            lora_list.append(lora)
            model_id = self.lora_list[lora_id].split("/")[3:5]
            model_url_list.append(f"https://www.modelscope.cn/models/{model_id[0]}/{model_id[1]}")
        return lora_list, model_url_list



model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev"])
pipe = FluxImagePipeline.from_model_manager(model_manager)
pipe.enable_auto_lora()

lora_patcher = LoraPatcher().to(dtype=torch.bfloat16, device="cuda")
lora_patcher.load_state_dict(load_state_dict("models/lora_merger/epoch-9.safetensors"))

retriever = LoRARetrieverTrainingModel("models/lora_retriever/epoch-3.safetensors").to(dtype=torch.bfloat16, device="cuda")
retriever.process_lora_list(list(set("data/lora/models/" + i for i in pd.read_csv("data/lora/lora_dataset_1000.csv")["model_file"])))

dataset = LoraDataset("data/lora/models", "data/lora/lora_dataset_1000.csv", steps_per_epoch=800, loras_per_item=1)

text_list = []
model_url_list = []
for seed in range(100):
    text = dataset[0][0]["text"]
    print(text)
    loras, urls = retriever.retrieve(text, k=3)
    print(urls)
    image = pipe(
        prompt=text,
        seed=seed,
    )
    image.save(f"data/lora/lora_outputs/image_{seed}_top0.jpg")
    for i in range(2, 3):
        image = pipe(
            prompt=text,
            lora_state_dicts=loras[:i+1],
            lora_patcher=lora_patcher,
            seed=seed,
        )
        image.save(f"data/lora/lora_outputs/image_{seed}_top{i+1}.jpg")
        
    text_list.append(text)
    model_url_list.append(urls)
    df = pd.DataFrame()
    df["text"] = text_list
    df["models"] = [",".join(i) for i in model_url_list]
    df.to_csv("data/lora/lora_outputs/metadata.csv", index=False, encoding="utf-8-sig")