import torch
import pandas as pd
from PIL import Image
import lightning as pl
from diffsynth import ModelManager, FluxImagePipeline, download_models, load_state_dict
from diffsynth.models.lora import LoRAFromCivitai, FluxLoRAConverter
from diffsynth.data.video import crop_and_resize
from diffsynth.pipelines.flux_image import lets_dance_flux
from torchvision.transforms import v2



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



class LightningModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-4,
        use_gradient_checkpointing=True,
        state_dict_converter=FluxLoRAConverter.align_to_diffsynth_format,
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device=self.device, model_id_list=["FLUX.1-dev"])
        self.pipe = FluxImagePipeline.from_model_manager(model_manager)
        self.lora_patcher = LoraPatcher()
        self.pipe.enable_auto_lora()
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        # Set parameters
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.state_dict_converter = state_dict_converter


    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()


    def training_step(self, batch, batch_idx):
        # Data
        text, image = batch["text"], batch["image"]
        lora_state_dicts = [
            self.state_dict_converter(load_state_dict(batch["model_file"][0], torch_dtype=torch.bfloat16, device=self.device)),
            self.state_dict_converter(load_state_dict(batch["model_file_extra"][0], torch_dtype=torch.bfloat16, device=self.device)),
        ]
        lora_alpahs = [1, 1]

        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt(text, positive=True)
        if "latents" in batch:
            latents = batch["latents"].to(dtype=self.pipe.torch_dtype, device=self.device)
        else:
            latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = lets_dance_flux(
            self.pipe.dit,
            hidden_states=noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
            lora_state_dicts=lora_state_dicts, lora_alpahs=lora_alpahs, lora_patcher=self.lora_patcher,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.lora_patcher.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        checkpoint.update(self.lora_patcher.state_dict())


if __name__ == '__main__':
    model = LightningModel(learning_rate=1e-4)
    dataset = LoraDataset("data/loras.csv", steps_per_epoch=500)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)
    trainer = pl.Trainer(
        max_epochs=100000,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy="auto",
        default_root_dir="./models",
        accumulate_grad_batches=1,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
