import torch, json, os, imageio
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
from diffsynth import ModelManager, EnhancedDDIMScheduler, SDVideoPipeline, SDUNet, load_state_dict, SDMotionModel



def lets_dance(
    unet: SDUNet,
    motion_modules: SDMotionModel,
    sample,
    timestep,
    encoder_hidden_states,
    use_gradient_checkpointing=False,
):
    # 1. ControlNet (skip)
    # 2. time
    time_emb = unet.time_proj(timestep[None]).to(sample.dtype)
    time_emb = unet.time_embedding(time_emb)

    # 3. pre-process
    hidden_states = unet.conv_in(sample)
    text_emb = encoder_hidden_states
    res_stack = [hidden_states]

    # 4. blocks
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward
    for block_id, block in enumerate(unet.blocks):
        # 4.1 UNet
        if use_gradient_checkpointing:
            hidden_states, time_emb, text_emb, res_stack = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states, time_emb, text_emb, res_stack,
                use_reentrant=False,
            )
        else:
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)
        # 4.2 AnimateDiff
        if block_id in motion_modules.call_block_id:
            motion_module_id = motion_modules.call_block_id[block_id]
            if use_gradient_checkpointing:
                hidden_states, time_emb, text_emb, res_stack = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(motion_modules.motion_modules[motion_module_id]),
                    hidden_states, time_emb, text_emb, res_stack,
                    use_reentrant=False,
                )
            else:
                hidden_states, time_emb, text_emb, res_stack = motion_modules.motion_modules[motion_module_id](hidden_states, time_emb, text_emb, res_stack)
    
    # 5. output
    hidden_states = unet.conv_norm_out(hidden_states)
    hidden_states = unet.conv_act(hidden_states)
    hidden_states = unet.conv_out(hidden_states)

    return hidden_states



class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch=10000, training_shapes=[(128, 1, 128, 512, 512)]):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.path = [os.path.join(base_path, i["path"]) for i in metadata]
        self.text = [i["text"] for i in metadata]
        self.steps_per_epoch = steps_per_epoch
        self.training_shapes = training_shapes

        self.frame_process = []
        for max_num_frames, interval, num_frames, height, width in training_shapes:
            self.frame_process.append(v2.Compose([
                v2.Resize(size=max(height, width), antialias=True),
                v2.CenterCrop(size=(height, width)),
                v2.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
            ]))


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = torch.tensor(frame, dtype=torch.float32)
            frame = rearrange(frame, "H W C -> 1 C H W")
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.concat(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames


    def load_video(self, file_path, training_shape_id):
        data = {}
        max_num_frames, interval, num_frames, height, width = self.training_shapes[training_shape_id]
        frame_process = self.frame_process[training_shape_id]
        start_frame_id = torch.randint(0, max_num_frames - (num_frames - 1) * interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process)
        if frames is None:
            return None
        else:
            data[f"frames_{training_shape_id}"] = frames
            data[f"start_frame_id_{training_shape_id}"] = start_frame_id
        return data


    def __getitem__(self, index):
        video_data = {}
        for training_shape_id in range(len(self.training_shapes)):
            while True:
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path) # For fixed seed.
                text = self.text[data_id]
                if isinstance(text, list):
                    text = text[torch.randint(0, len(text), (1,))[0]]
                video_file = self.path[data_id]
                try:
                    data = self.load_video(video_file, training_shape_id)
                except:
                    data = None
                if data is not None:
                    data[f"text_{training_shape_id}"] = text
                    break
            video_data.update(data)
        return video_data
    

    def __len__(self):
        return self.steps_per_epoch



class LightningModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, sd_ckpt_path=None):
        super().__init__()
        # Load models
        model_manager = ModelManager(torch_dtype=torch.float16, device="cpu")
        model_manager.load_stable_diffusion(load_state_dict(sd_ckpt_path))
        
        # Initialize motion modules
        model_manager.model["motion_modules"] = SDMotionModel().to(dtype=self.dtype, device=self.device)

        # Build pipeline
        self.pipe = SDVideoPipeline.from_model_manager(model_manager)
        self.pipe.vae_encoder.eval()
        self.pipe.vae_encoder.requires_grad_(False)

        self.pipe.vae_decoder.eval()
        self.pipe.vae_decoder.requires_grad_(False)

        self.pipe.text_encoder.eval()
        self.pipe.text_encoder.requires_grad_(False)

        self.pipe.unet.eval()
        self.pipe.unet.requires_grad_(False)

        self.pipe.motion_modules.train()
        self.pipe.motion_modules.requires_grad_(True)

        # Reset the scheduler
        self.pipe.scheduler = EnhancedDDIMScheduler(beta_schedule="scaled_linear")
        self.pipe.scheduler.set_timesteps(1000)

        # Other parameters
        self.learning_rate = learning_rate


    def encode_video_with_vae(self, video):
        video = video.to(device=self.device, dtype=self.dtype)
        video = video.unsqueeze(0)
        latents = self.pipe.vae_encoder.encode_video(video, batch_size=16)
        latents = rearrange(latents[0], "C T H W -> T C H W")
        return latents
        

    def calculate_loss(self, prompt, frames):
        with torch.no_grad():
            # Call video encoder
            latents = self.encode_video_with_vae(frames)

            # Call text encoder
            prompt_embs = self.pipe.prompter.encode_prompt(self.pipe.text_encoder, prompt, device=self.device, max_length=77)
            prompt_embs = prompt_embs.repeat(latents.shape[0], 1, 1)

            # Call scheduler
            timestep = torch.randint(0, len(self.pipe.scheduler.timesteps), (1,), device=self.device)[0]
            noise = torch.randn_like(latents)
            noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)

        # Calculate loss
        model_pred = lets_dance(
            self.pipe.unet, self.pipe.motion_modules,
            sample=noisy_latents, encoder_hidden_states=prompt_embs, timestep=timestep
        )
        loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        return loss
    

    def training_step(self, batch, batch_idx):
        # Loss
        frames = batch["frames_0"][0]
        prompt = batch["text_0"][0]
        loss = self.calculate_loss(prompt, frames)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.pipe.motion_modules.parameters(), lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.motion_modules.named_parameters()))
        trainable_param_names = [named_param[0] for named_param in trainable_param_names]
        checkpoint["trainable_param_names"] = trainable_param_names



if __name__ == '__main__':
    # dataset and data loader
    dataset = TextVideoDataset(
        "/data/zhongjie/datasets/opensoraplan/data/processed",
        "/data/zhongjie/datasets/opensoraplan/data/processed/metadata.json",
        training_shapes=[(16, 1, 16, 512, 512)],
        steps_per_epoch=7*10000,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=4
    )

    # model
    model = LightningModel(
        learning_rate=1e-5,
        sd_ckpt_path="models/stable_diffusion/v1-5-pruned-emaonly.safetensors",
    )

    # train
    trainer = pl.Trainer(
        max_epochs=100000,
        accelerator="gpu",
        devices="auto",
        strategy="deepspeed_stage_1",
        precision="16-mixed",
        default_root_dir="/data/zhongjie/models/train_extended_animatediff",
        accumulate_grad_batches=1,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)]
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        ckpt_path=None
    )
