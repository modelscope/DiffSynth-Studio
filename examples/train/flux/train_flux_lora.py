from diffsynth import ModelManager, FluxImagePipeline
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
from diffsynth.models.lora import FluxLoRAConverter
import torch, os, argparse
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
        
        if quantize is not None:
            self.pipe.dit.quantize()
        
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.freeze_parameters()
        self.add_lora_to_model(
            self.pipe.denoising_model(),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            init_lora_weights=init_lora_weights,
            pretrained_lora_path=pretrained_lora_path,
            state_dict_converter=FluxLoRAConverter.align_to_diffsynth_format
        )


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
    launch_training_task(model, args)
