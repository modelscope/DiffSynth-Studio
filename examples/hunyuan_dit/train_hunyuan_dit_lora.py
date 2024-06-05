from diffsynth import ModelManager, HunyuanDiTImagePipeline
from peft import LoraConfig, inject_adapter_in_model
from torchvision import transforms
from PIL import Image
import lightning as pl
import pandas as pd
import torch, os, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "True"



class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False):
        self.steps_per_epoch = steps_per_epoch
        metadata = pd.read_csv(os.path.join(dataset_path, "train/metadata.csv"))
        self.path = [os.path.join(dataset_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        self.image_processor = transforms.Compose(
            [
                transforms.Resize(max(height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        text = self.text[data_id]
        image = Image.open(self.path[data_id]).convert("RGB")
        image = self.image_processor(image)
        return {"text": text, "image": image}
    

    def __len__(self):
        return self.steps_per_epoch
    


class LightningModel(pl.LightningModule):
    def __init__(self, torch_dtype=torch.float16, learning_rate=1e-4, pretrained_weights=[], lora_rank=4, lora_alpha=4, use_gradient_checkpointing=True):
        super().__init__()

        # Load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        model_manager.load_models(pretrained_weights)
        self.pipe = HunyuanDiTImagePipeline.from_model_manager(model_manager)

        # Freeze parameters
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_t5.requires_grad_(False)
        self.pipe.dit.requires_grad_(False)
        self.pipe.vae_decoder.requires_grad_(False)
        self.pipe.vae_encoder.requires_grad_(False)
        self.pipe.text_encoder.eval()
        self.pipe.text_encoder_t5.eval()
        self.pipe.dit.train()
        self.pipe.vae_decoder.eval()
        self.pipe.vae_encoder.eval()

        # Add LoRA to DiT
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out"],
        )
        self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)
        for param in self.pipe.dit.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

        # Set other parameters
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
    

    def training_step(self, batch, batch_idx):
        # Data
        text, image = batch["text"], batch["image"]

        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb, attention_mask, prompt_emb_t5, attention_mask_t5 = self.pipe.prompter.encode_prompt(
            self.pipe.text_encoder, self.pipe.text_encoder_t5, text, positive=True, device=self.device
        )
        latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))
        noise = torch.randn_like(latents)
        timestep = torch.randint(0, 1000, (1,), device=self.device)
        extra_input = self.pipe.prepare_extra_input(image.shape[-2], image.shape[-1], batch_size=latents.shape[0])
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.dit(
            noisy_latents,
            prompt_emb, prompt_emb_t5, attention_mask, attention_mask_t5,
            timestep,
            **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        loss = torch.nn.functional.mse_loss(noise_pred, training_target)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.dit.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.dit.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.dit.state_dict()
        for name, param in state_dict.items():
            if name in trainable_param_names:
                checkpoint[name] = param



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model. For example, `./HunyuanDiT/t2i`.",
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
        "--height",
        type=int,
        default=1024,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="Whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16", "16-mixed"],
        help="Training precision",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # args
    args = parse_args()

    # dataset and data loader
    dataset = TextImageDataset(
        args.dataset_path,
        steps_per_epoch=args.steps_per_epoch * args.batch_size,
        height=args.height,
        width=args.width,
        center_crop=args.center_crop,
        random_flip=args.random_flip
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )

    # model
    model = LightningModel(
        pretrained_weights=[
            os.path.join(args.pretrained_path, "clip_text_encoder/pytorch_model.bin"),
            os.path.join(args.pretrained_path, "mt5/pytorch_model.bin"),
            os.path.join(args.pretrained_path, "model/pytorch_model_ema.pt"),
            os.path.join(args.pretrained_path, "sdxl-vae-fp16-fix/diffusion_pytorch_model.bin"),
        ],
        torch_dtype=torch.float32 if args.precision == "32" else torch.float16,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing=args.use_gradient_checkpointing
    )

    # train
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision=args.precision,
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)]
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
