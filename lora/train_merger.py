from diffsynth import FluxImagePipeline, ModelManager
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.pipelines.flux_image import lets_dance_flux
from lora.dataset import LoraDataset
from lora.merger import LoraPatcher
from lora.utils import load_lora
import torch, os
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm



class LoRAMergerTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu", model_id_list=["FLUX.1-dev"])
        self.pipe = FluxImagePipeline.from_model_manager(model_manager)
        self.lora_patcher = LoraPatcher()
        self.pipe.enable_auto_lora()
        self.freeze_parameters()
        self.switch_to_training_mode()
        self.use_gradient_checkpointing = True
        self.state_dict_converter = FluxLoRAConverter.align_to_diffsynth_format
        self.device = "cuda"
        
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self
        
        
    def switch_to_training_mode(self):
        self.pipe.scheduler.set_timesteps(1000, training=True)


    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        self.lora_patcher.requires_grad_(True)


    def forward(self, batch):
        # Data
        text, image = batch[0]["text"], batch[0]["image"].unsqueeze(0)
        num_lora = torch.randint(1, len(batch), (1,))[0]
        lora_state_dicts = [
            self.state_dict_converter(load_lora(batch[i]["model_file"], device=self.device)) for i in range(num_lora)
        ]
        lora_alphas = None

        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt(text, positive=True)
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
            lora_state_dicts=lora_state_dicts, lora_alphas=lora_alphas, lora_patcher=self.lora_patcher,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        return loss
    
    
    def trainable_modules(self):
        return self.lora_patcher.parameters()


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        
    
    def on_step_end(self, loss):
        pass
    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).lora_patcher.state_dict()
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


if __name__ == '__main__':
    model = LoRAMergerTrainingModel()
    dataset = LoraDataset("data/lora/models/", "data/lora/lora_dataset_1000.csv", steps_per_epoch=800, loras_per_item=4)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1, collate_fn=lambda x: x[0])
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=1e-4)
    model_logger = ModelLogger("models/lora_merger")
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    for epoch_id in range(1000000):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
        model_logger.on_epoch_end(accelerator, model, epoch_id)
