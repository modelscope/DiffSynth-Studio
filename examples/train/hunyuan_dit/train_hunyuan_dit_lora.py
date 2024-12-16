from diffsynth import ModelManager, HunyuanDiTImagePipeline
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
        self.pipe = HunyuanDiTImagePipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000)

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
        "--pretrained_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model. For example, `./HunyuanDiT/t2i`.",
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
            os.path.join(args.pretrained_path, "clip_text_encoder/pytorch_model.bin"),
            os.path.join(args.pretrained_path, "mt5/pytorch_model.bin"),
            os.path.join(args.pretrained_path, "model/pytorch_model_ema.pt"),
            os.path.join(args.pretrained_path, "sdxl-vae-fp16-fix/diffusion_pytorch_model.bin"),
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
