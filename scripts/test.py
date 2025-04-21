import torch, shutil, os
from diffsynth import ModelManager, FluxImagePipeline, download_models, load_state_dict
from diffsynth.models.lora import LoRAFromCivitai, FluxLoRAConverter
import pandas as pd
import torch
import pandas as pd
from PIL import Image
import lightning as pl
from diffsynth import ModelManager, FluxImagePipeline, download_models, load_state_dict
from diffsynth.models.lora import LoRAFromCivitai, FluxLoRAConverter
from diffsynth.data.video import crop_and_resize
from diffsynth.pipelines.flux_image import lets_dance_flux
from torchvision.transforms import v2


baseline = "trained"


class LoraMerger(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight_base = torch.nn.Parameter(torch.randn((dim,)))
        self.weight_lora = torch.nn.Parameter(torch.randn((dim,)))
        self.weight_cross = torch.nn.Parameter(torch.randn((dim,)))
        self.weight_out = torch.nn.Parameter(torch.ones((dim,)))
        self.bias = torch.nn.Parameter(torch.randn((dim,)))
        self.activation = torch.nn.Sigmoid()
        self.norm_base = torch.nn.LayerNorm(dim, eps=1e-5)
        self.norm_lora = torch.nn.LayerNorm(dim, eps=1e-5)
        
    def forward(self, base_output, lora_outputs):
        global baseline
        if baseline == "nolora":
            output = base_output
        elif baseline == "lora1":
            output = base_output + lora_outputs[0]
        elif baseline == "lora2":
            output = base_output + lora_outputs[1]
        elif baseline == "alllora":
            output = base_output + lora_outputs.sum(dim=0)
        else:
            norm_base_output = self.norm_base(base_output)
            norm_lora_outputs = self.norm_lora(lora_outputs)
            gate = self.activation(
                norm_base_output * self.weight_base \
                + norm_lora_outputs * self.weight_lora \
                + norm_base_output * norm_lora_outputs * self.weight_cross + self.bias
            )
            output = base_output + (self.weight_out * gate * lora_outputs).sum(dim=0)
        return output



class LoraPatcher(torch.nn.Module):
    def __init__(self, lora_patterns=None):
        super().__init__()
        if lora_patterns is None:
            lora_patterns = self.default_lora_patterns()
        model_dict = {}
        for lora_pattern in lora_patterns:
            name, dim = lora_pattern["name"], lora_pattern["dim"]
            model_dict[name.replace(".", "___")] = LoraMerger(dim)
        self.model_dict = torch.nn.ModuleDict(model_dict)
        
    def default_lora_patterns(self):
        lora_patterns = []
        lora_dict = {
            "attn.a_to_qkv": 9216, "attn.a_to_out": 3072, "ff_a.0": 12288, "ff_a.2": 3072, "norm1_a.linear": 18432,
            "attn.b_to_qkv": 9216, "attn.b_to_out": 3072, "ff_b.0": 12288, "ff_b.2": 3072, "norm1_b.linear": 18432,
        }
        for i in range(19):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix]
                })
        lora_dict = {"to_qkv_mlp": 21504, "proj_out": 3072, "norm.linear": 9216}
        for i in range(38):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"single_blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix]
                })
        return lora_patterns
        
    def forward(self, base_output, lora_outputs, name):
        return self.model_dict[name.replace(".", "___")](base_output, lora_outputs)


class LoraDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, steps_per_epoch=1000):
        data_df = pd.read_csv(metadata_path)
        self.model_file = data_df["model_file"].tolist()
        self.image_file = data_df["image_file"].tolist()
        self.text = data_df["text"].tolist()
        self.max_resolution = 1920 * 1080
        self.steps_per_epoch = steps_per_epoch
        
        
    def read_image(self, image_file):
        image = Image.open(image_file)
        width, height = image.size
        if width * height > self.max_resolution:
            scale = (width * height / self.max_resolution) ** 0.5
            image = image.resize((int(width / scale), int(height / scale)))
            width, height = image.size
        if width % 16 != 0 or height % 16 != 0:
            image = crop_and_resize(image, height // 16 * 16, width // 16 * 16)
        image = v2.functional.to_image(image)
        image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)
        image = v2.functional.normalize(image, [0.5], [0.5])
        return image


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.model_file), (1,))[0]
        data_id = (data_id + index) % len(self.model_file) # For fixed seed.
        data_id_extra = torch.randint(0, len(self.model_file), (1,))[0]
        return {
            "model_file": self.model_file[data_id],
            "model_file_extra": self.model_file[data_id_extra],
            "image": self.read_image(self.image_file[data_id]),
            "text": self.text[data_id]
        }


    def __len__(self):
        return self.steps_per_epoch
    



model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
pipe = FluxImagePipeline.from_model_manager(model_manager)
pipe.enable_auto_lora()
    

lora_alpahs = [1, 1]
lora_patcher = LoraPatcher().to(dtype=torch.bfloat16, device="cuda")
lora_patcher.load_state_dict(load_state_dict("models/lightning_logs/version_13/checkpoints/epoch=2-step=1500.ckpt"))

dataset = LoraDataset("data/loras_picked.csv")

for seed in range(100):
    data = dataset[0]
    lora_state_dicts = [
        FluxLoRAConverter.align_to_diffsynth_format(load_state_dict(data["model_file"], torch_dtype=torch.bfloat16, device="cuda")),
        FluxLoRAConverter.align_to_diffsynth_format(load_state_dict(data["model_file_extra"], torch_dtype=torch.bfloat16, device="cuda")),
    ]
    lora_alpahs = [1, 1]
    for pattern in ["nolora", "lora1", "lora2", "alllora", "loramerger"]:
        baseline = pattern
        image = pipe(
            prompt=data["text"],
            lora_state_dicts=lora_state_dicts, 
            lora_alpahs=lora_alpahs,
            lora_patcher=lora_patcher,
            seed=seed,
        )
        image.save(f"data/lora_outputs/image_{seed}_{pattern}.jpg")