from diffsynth import ModelManager, FluxImagePipeline
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
from diffsynth.models.lora import FluxLoRAConverter
import torch, os, argparse
import lightning as pl
from diffsynth.data.image_pulse import SingleTaskDataset, MultiTaskDataset
from diffsynth.pipelines.flux_image import lets_dance_flux
from diffsynth.models.flux_reference_embedder import FluxReferenceEmbedder
os.environ["TOKENIZERS_PARALLELISM"] = "True"


class LightningModel(LightningModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[], preset_lora_path=None,
        learning_rate=1e-4, use_gradient_checkpointing=True,
        lora_rank=4, lora_alpha=4, lora_target_modules="to_q,to_k,to_v,to_out", init_lora_weights="kaiming", pretrained_lora_path=None,
        state_dict_converter=None, quantize = None
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing, state_dict_converter=state_dict_converter)
        # Load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        if quantize is None:
            model_manager.load_models(pretrained_weights)
        else:
            model_manager.load_models(pretrained_weights[1:])
            model_manager.load_model(pretrained_weights[0], torch_dtype=quantize)
        if preset_lora_path is not None:
            preset_lora_path = preset_lora_path.split(",")
            for path in preset_lora_path:
                model_manager.load_lora(path)
            
        self.pipe = FluxImagePipeline.from_model_manager(model_manager)
        self.pipe.reference_embedder = FluxReferenceEmbedder()
        
        if quantize is not None:
            self.pipe.dit.quantize()
        
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.freeze_parameters()
        self.pipe.reference_embedder.requires_grad_(True)
        self.pipe.reference_embedder.train()
        # self.add_lora_to_model(
        #     self.pipe.denoising_model(),
        #     lora_rank=lora_rank,
        #     lora_alpha=lora_alpha,
        #     lora_target_modules=lora_target_modules,
        #     init_lora_weights=init_lora_weights,
        #     pretrained_lora_path=pretrained_lora_path,
        #     state_dict_converter=FluxLoRAConverter.align_to_diffsynth_format
        # )
        
        
    def training_step(self, batch, batch_idx):
        # Data
        text, image = batch["instruction"], batch["image_2"]
        image_ref = batch["image_1"]

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
        
        # Reference image
        hidden_states_ref = self.pipe.vae_encoder(image_ref.to(dtype=self.pipe.torch_dtype, device=self.device))

        # Compute loss
        noise_pred = lets_dance_flux(
            self.pipe.denoising_model(),
            reference_embedder=self.pipe.reference_embedder,
            hidden_states_ref=hidden_states_ref,
            hidden_states=noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    
    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        if self.state_dict_converter is not None:
            lora_state_dict = self.state_dict_converter(lora_state_dict, alpha=self.lora_alpha)
        checkpoint.update(lora_state_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_text_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained text encoder model. For example, `models/FLUX/FLUX.1-dev/text_encoder/model.safetensors`.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_2_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained t5 text encoder model. For example, `models/FLUX/FLUX.1-dev/text_encoder_2`.",
    )
    parser.add_argument(
        "--pretrained_dit_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained dit model. For example, `models/FLUX/FLUX.1-dev/flux1-dev.safetensors`.",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained vae model. For example, `models/FLUX/FLUX.1-dev/ae.safetensors`.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--align_to_opensource_format",
        default=False,
        action="store_true",
        help="Whether to export lora files aligned with other opensource format.",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["float8_e4m3fn"],
        help="Whether to use quantization when training the model, and in which format.",
    )
    parser.add_argument(
        "--preset_lora_path",
        type=str,
        default=None,
        help="Preset LoRA path.",
    )
    parser = add_general_parsers(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = LightningModel(
        torch_dtype={"32": torch.float32, "bf16": torch.bfloat16}.get(args.precision, torch.float16),
        pretrained_weights=[args.pretrained_dit_path, args.pretrained_text_encoder_path, args.pretrained_text_encoder_2_path, args.pretrained_vae_path],
        preset_lora_path=args.preset_lora_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        pretrained_lora_path=args.pretrained_lora_path,
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format if args.align_to_opensource_format else None,
        quantize={"float8_e4m3fn": torch.float8_e4m3fn}.get(args.quantize, None),
    )
    # dataset and data loader
    dataset = MultiTaskDataset(
        dataset_list=[
            SingleTaskDataset(
                "/shark/zhongjie/data/image_pulse_datasets/task1/data/dataset_change_add_remove",
                metadata_path="/shark/zhongjie/data/image_pulse_datasets/task1/data/metadata/20250411_dataset_change_add_remove.json",
                height=512, width=512,
            ),
            SingleTaskDataset(
                "/shark/zhongjie/data/image_pulse_datasets/task1/data/dataset_zoomin_zoomout",
                metadata_path="/shark/zhongjie/data/image_pulse_datasets/task1/data/metadata/20250411_dataset_zoomin_zoomout.json",
                height=512, width=512,
            ),
            SingleTaskDataset(
                "/shark/zhongjie/data/image_pulse_datasets/task1/data/dataset_style_transfer",
                keys=(("image_1", "image_4", "editing_instruction"), ("image_4", "image_1", "reverse_editing_instruction")),
                metadata_path="/shark/zhongjie/data/image_pulse_datasets/task1/data/metadata/20250411_dataset_style_transfer.json",
                height=512, width=512,
            ),
            SingleTaskDataset(
                "/shark/zhongjie/data/image_pulse_datasets/task1/data/dataset_faceid",
                metadata_path="/shark/zhongjie/data/image_pulse_datasets/task1/data/metadata/20250411_dataset_faceid.json",
                height=512, width=512,
            ),
        ],
        dataset_weight=(4, 2, 2, 1),
        steps_per_epoch=args.steps_per_epoch,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
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
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=None,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
