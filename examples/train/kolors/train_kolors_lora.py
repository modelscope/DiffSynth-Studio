from diffsynth import ModelManager, SDXLImagePipeline
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
import torch, os, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "True"


class LightningModel(LightningModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[],
        learning_rate=1e-4, use_gradient_checkpointing=True,
        lora_rank=4, lora_alpha=4, lora_target_modules="to_q,to_k,to_v,to_out", init_lora_weights="gaussian", pretrained_lora_path=None,
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        # Load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        model_manager.load_models(pretrained_weights)
        self.pipe = SDXLImagePipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1100)

        # Convert the vae encoder to torch.float16
        self.pipe.vae_encoder.to(torch_dtype)

        self.freeze_parameters()
        self.add_lora_to_model(
            self.pipe.denoising_model(),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            init_lora_weights=init_lora_weights,
            pretrained_lora_path=pretrained_lora_path,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_unet_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model (UNet). For example, `models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors`.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model (Text Encoder). For example, `models/kolors/Kolors/text_encoder`.",
    )
    parser.add_argument(
        "--pretrained_fp16_vae_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model (VAE). For example, `models/kolors/Kolors/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors`.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="to_q,to_k,to_v,to_out",
        help="Layers with LoRA modules.",
    )
    parser = add_general_parsers(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = LightningModel(
        torch_dtype=torch.float32 if args.precision == "32" else torch.float16,
        pretrained_weights=[
            args.pretrained_unet_path,
            args.pretrained_text_encoder_path,
            args.pretrained_fp16_vae_path,
        ],
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=args.init_lora_weights,
        pretrained_lora_path=args.pretrained_lora_path,
        lora_target_modules=args.lora_target_modules
    )
    launch_training_task(model, args)
