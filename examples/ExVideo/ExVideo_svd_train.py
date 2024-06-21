import torch, json, os, imageio, argparse
from torchvision.transforms import v2
import numpy as np
from einops import rearrange, repeat
import lightning as pl
from diffsynth import ModelManager, SVDImageEncoder, SVDUNet, SVDVAEEncoder, ContinuousODEScheduler, load_state_dict
from diffsynth.pipelines.stable_video_diffusion import SVDCLIPImageProcessor
from diffsynth.models.svd_unet import TemporalAttentionBlock



class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch=10000, training_shapes=[(128, 1, 128, 512, 512)]):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.path = [os.path.join(base_path, i["path"]) for i in metadata]
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
        return data


    def __getitem__(self, index):
        video_data = {}
        for training_shape_id in range(len(self.training_shapes)):
            while True:
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path) # For fixed seed.
                video_file = self.path[data_id]
                try:
                    data = self.load_video(video_file, training_shape_id)
                except:
                    data = None
                if data is not None:
                    break
            video_data.update(data)
        return video_data
    

    def __len__(self):
        return self.steps_per_epoch



class MotionBucketManager:
    def __init__(self):
        self.thresholds = [
            0.000000000, 0.012205946, 0.015117834, 0.018080613, 0.020614484, 0.021959992, 0.024088068, 0.026323952, 
            0.028277775, 0.029968588, 0.031836554, 0.033596724, 0.035121530, 0.037200287, 0.038914755, 0.040696491, 
            0.042368013, 0.044265781, 0.046311017, 0.048243891, 0.050294187, 0.052142400, 0.053634230, 0.055612389, 
            0.057594258, 0.059410289, 0.061283995, 0.063603796, 0.065192916, 0.067146860, 0.069066539, 0.070390493, 
            0.072588451, 0.073959745, 0.075889029, 0.077695683, 0.079783581, 0.082162730, 0.084092639, 0.085958421, 
            0.087700523, 0.089684933, 0.091688842, 0.093335517, 0.094987206, 0.096664011, 0.098314710, 0.100262381, 
            0.101984538, 0.103404313, 0.105280340, 0.106974818, 0.109028399, 0.111164779, 0.113065213, 0.114362158, 
            0.116407216, 0.118063427, 0.119524263, 0.121835820, 0.124242283, 0.126202747, 0.128989249, 0.131672353, 
            0.133417681, 0.135567948, 0.137313649, 0.139189199, 0.140912935, 0.143525436, 0.145718485, 0.148315132, 
            0.151039496, 0.153218940, 0.155252382, 0.157651082, 0.159966752, 0.162195817, 0.164811596, 0.167341709, 
            0.170251891, 0.172651157, 0.175550997, 0.178372145, 0.181039348, 0.183565900, 0.186599866, 0.190071866, 
            0.192574754, 0.195026234, 0.198099136, 0.200210452, 0.202522039, 0.205410406, 0.208610669, 0.211623028, 
            0.214723110, 0.218520239, 0.222194016, 0.225363150, 0.229384825, 0.233422622, 0.237012610, 0.240735114, 
            0.243622541, 0.247465774, 0.252190471, 0.257356376, 0.261856794, 0.266556412, 0.271076709, 0.277361482, 
            0.281250387, 0.286582440, 0.291158527, 0.296712339, 0.303008437, 0.311793238, 0.318485111, 0.326999635, 
            0.332138240, 0.341770738, 0.354188830, 0.365194678, 0.379234344, 0.401538879, 0.416078776, 0.440871328,
        ]

    def get_motion_score(self, frames):
        score = frames.std(dim=2).mean(dim=[1, 2, 3]).tolist()
        return score
    
    def get_bucket_id(self, motion_score):
        for bucket_id in range(len(self.thresholds) - 1):
            if self.thresholds[bucket_id + 1] > motion_score:
                return bucket_id
        return len(self.thresholds) - 1

    def __call__(self, frames):
        scores = self.get_motion_score(frames)
        bucket_ids = [self.get_bucket_id(score) for score in scores]
        return bucket_ids



class LightningModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, svd_ckpt_path=None, add_positional_conv=128, contrast_enhance_scale=1.01):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.float16, device=self.device)
        model_manager.load_stable_video_diffusion(state_dict=load_state_dict(svd_ckpt_path), add_positional_conv=add_positional_conv)

        self.image_encoder: SVDImageEncoder = model_manager.image_encoder
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)

        self.unet: SVDUNet = model_manager.unet
        self.unet.train()
        self.unet.requires_grad_(False)
        for block in self.unet.blocks:
            if isinstance(block, TemporalAttentionBlock):
                block.requires_grad_(True)

        self.vae_encoder: SVDVAEEncoder = model_manager.vae_encoder
        self.vae_encoder.eval()
        self.vae_encoder.requires_grad_(False)

        self.noise_scheduler = ContinuousODEScheduler(num_inference_steps=1000)
        self.learning_rate = learning_rate

        self.motion_bucket_manager = MotionBucketManager()
        self.contrast_enhance_scale = contrast_enhance_scale


    def encode_image_with_clip(self, image):
        image = SVDCLIPImageProcessor().resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1).to(device=self.device, dtype=self.dtype)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1).to(device=self.device, dtype=self.dtype)
        image = (image - mean) / std
        image_emb = self.image_encoder(image)
        return image_emb
    

    def encode_video_with_vae(self, video):
        video = video.to(device=self.device, dtype=self.dtype)
        video = video.unsqueeze(0)
        latents = self.vae_encoder.encode_video(video)
        latents = rearrange(latents[0], "C T H W -> T C H W")
        return latents
    

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        return frames


    def calculate_loss(self, frames):
        with torch.no_grad():
            # Call video encoder
            latents = self.encode_video_with_vae(frames)
            image_emb_vae = repeat(latents[0] / self.vae_encoder.scaling_factor, "C H W -> T C H W", T=frames.shape[1])
            image_emb_clip = self.encode_image_with_clip(frames[:,0].unsqueeze(0))

            # Call scheduler
            timestep = torch.randint(0, len(self.noise_scheduler.timesteps), (1,))[0]
            timestep = self.noise_scheduler.timesteps[timestep]
            noise = torch.randn_like(latents)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timestep)

            # Prepare positional id
            fps = 30
            motion_bucket_id = self.motion_bucket_manager(frames.unsqueeze(0))[0]
            noise_aug_strength = 0
            add_time_id = torch.tensor([[fps-1, motion_bucket_id, noise_aug_strength]], device=self.device)

        # Calculate loss
        latents_input = torch.cat([noisy_latents, image_emb_vae], dim=1)
        model_pred = self.unet(latents_input, timestep, image_emb_clip, add_time_id, use_gradient_checkpointing=True)
        latents_output = self.noise_scheduler.step(model_pred.float(), timestep, noisy_latents.float(), to_final=True)
        loss = torch.nn.functional.mse_loss(latents_output, latents.float() * self.contrast_enhance_scale, reduction="mean")

        # Re-weighting
        reweighted_loss = loss * self.noise_scheduler.training_weight(timestep)
        return loss, reweighted_loss
    

    def training_step(self, batch, batch_idx):
        # Loss
        frames = batch["frames_0"][0]
        loss, reweighted_loss = self.calculate_loss(frames)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        self.log("reweighted_train_loss", reweighted_loss, prog_bar=True)
        return reweighted_loss


    def configure_optimizers(self):
        trainable_modules = []
        for block in self.unet.blocks:
            if isinstance(block, TemporalAttentionBlock):
                trainable_modules += block.parameters()
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.unet.named_parameters()))
        trainable_param_names = [named_param[0] for named_param in trainable_param_names]
        checkpoint["trainable_param_names"] = trainable_param_names



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model. For example, `models/stable_video_diffusion/svd_xt.safetensors`.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        required=False,
        help="Path to checkpoint, in case your training program is stopped unexpectedly and you want to resume.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=128,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--contrast_enhance_scale",
        type=float,
        default=1.01,
        help="Avoid generating gray videos.",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # args
    args = parse_args()

    # dataset and data loader
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.json"),
        training_shapes=[(args.num_frames, 1, args.num_frames, args.height, args.width)],
        steps_per_epoch=args.steps_per_epoch,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        # We don't support batch_size > 1,
        # because sometimes our GPU cannot process even one video.
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    # model
    model = LightningModel(
        learning_rate=args.learning_rate,
        svd_ckpt_path=args.pretrained_path,
        add_positional_conv=args.num_frames,
        contrast_enhance_scale=args.contrast_enhance_scale
    )

    # train
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        strategy="deepspeed_stage_2",
        precision="16-mixed",
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)]
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        ckpt_path=args.resume_from_checkpoint
    )
