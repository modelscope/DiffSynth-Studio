import argparse


def add_dataset_base_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    return parser

def add_image_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    return parser

def add_video_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    return parser

def add_model_config(parser: argparse.ArgumentParser):
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--extra_inputs", default=None, type=str, help="Additional model inputs, comma-separated.")
    parser.add_argument("--fp8_models", default=None, type=str, help="Models with FP8 precision, comma-separated.")
    parser.add_argument("--offload_models", default=None, type=str, help="Models with offload, comma-separated. Only used in splited training.")
    parser.add_argument("--resume_from_checkpoint", default=None, type=str, help="Resume training from checkpoint file. Only single model training is supported.")
    return parser

def add_quant_config(parser: argparse.ArgumentParser):
    parser.add_argument("--quant_options", type=str, default=None, help="Dynamically quantize loaded models. Semicolon-separated entries, each `<model_string>:<method>[/<exclude_modules>]`, where `<model_string>` matches an entry in `--model_paths`/`--model_id_with_origin_paths`, `method` is a registered method (e.g. bitsandbytes_nf4), and `exclude_modules` optionally lists layers kept in full precision. Alternatively, pass a JSON object mapping the same model strings to configs: `method`, optional `backend_config_kwargs` (method-specific parameters), and `target_modules`/`exclude_modules` (JSON lists of layer names). For example: --quant_options '{\"org/model:transformer/*.safetensors\": {\"method\": \"bitsandbytes_nf4\", \"backend_config_kwargs\": {\"blocksize\": 64}, \"exclude_modules\": [\"to_q\"]}}'. For mixed quantization, use a `configs` list of these configs instead of `method`, with `target_modules`/`exclude_modules` selecting non-overlapping layer sets.")
    return parser

def add_training_config(parser: argparse.ArgumentParser):
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    parser.add_argument("--customized_optimizer", type=str, default=None, help="Customized optimizer, e.g., `bitsandbytes.optim.Adam8bit` and `torch.optim.Adam`. The default optimizer is `torch.optim.AdamW`.")
    parser.add_argument("--warm_up_steps", type=int, default=5, help="Number of optimizer steps for linear learning rate warm-up. The learning rate ramps from `learning_rate / warm_up_steps` up to `learning_rate`. 0 disables warm-up. Steps are counted in optimizer updates, so the ramp is independent of `gradient_accumulation_steps` and the number of data-parallel processes.")
    return parser

def add_output_config(parser: argparse.ArgumentParser):
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    return parser

def add_lora_config(parser: argparse.ArgumentParser):
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--preset_lora_path", type=str, default=None, help="Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    parser.add_argument("--preset_lora_model", type=str, default=None, help="Which model the preset LoRA is fused to.")
    return parser

def add_gradient_config(parser: argparse.ArgumentParser):
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser

def add_template_model_config(parser: argparse.ArgumentParser):
    parser.add_argument("--template_model_id_or_path", type=str, default=None, help="Model ID of path of template models.")
    parser.add_argument("--enable_lora_hot_loading", default=False, action="store_true", help="Whether to enable LoRA hot-loading. Only available for image-to-lora models.")
    return parser

def add_offload_training_config(parser: argparse.ArgumentParser):
    parser.add_argument("--enable_model_cpu_offload", default=False, action="store_true", help="Enable layer offload training. Weights are kept on CPU and loaded to GPU one layer at a time.")
    parser.add_argument("--enable_optimizer_cpu_offload", default=False, action="store_true", help="When --enable_model_cpu_offload is enabled, run optimizer on CPU. All params are offloaded to CPU. Default is False (trainable params stay on GPU, optimizer on GPU).")
    parser.add_argument("--cpu_offload_split_threshold", type=int, default=None, help="Experimental! When --enable_model_cpu_offload is enabled, modules with total params above this threshold (in MB) are recursively split into children. None means offload every leaf module directly. Default: None.")
    return parser

def add_logger_config(parser: argparse.ArgumentParser):
    parser.add_argument("--enable_tensorboard_log", default=False, action="store_true", help="Enable tensorboard for logging.")
    parser.add_argument("--enable_swanlab_log", default=False, action="store_true", help="Enable swanlab for logging.")
    parser.add_argument("--swanlab_project", type=str, default="DiffSynth-Studio", help="SwanLab project name.")
    parser.add_argument("--enable_wandb_log", default=False, action="store_true", help="Enable wandb for logging.")
    parser.add_argument("--wandb_project", type=str, default="DiffSynth-Studio", help="Wandb project name.")
    parser.add_argument("--enable_csv_log", default=False, action="store_true", help="Enable CSV loss logging.")
    return parser

def add_dmd2_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dmd2_lora_dtype", choices=["bf16", "fp32"], default=None, help="LoRA parameter dtype; defaults to the pipeline dtype.")
    parser.add_argument("--dmd2_disable_gan", dest="dmd2_gan", action="store_false", default=True, help="Disable real-image GAN training.")
    parser.add_argument("--dmd2_fake_updates_per_generator", type=int, default=5)
    parser.add_argument("--dmd2_fake_warmup_steps", type=int, default=100, help="Initial critic-only updates before training the student; 0 disables fake warmup.")
    parser.add_argument("--dmd2_fake_learning_rate", type=float, default=None, help="Learning rate for fake and GAN classifier; defaults to learning_rate, which controls the student.")
    parser.add_argument("--dmd2_generator_gan_loss_weight", type=float, default=0.005, help="Weight of the GAN loss for generator updates.")
    parser.add_argument("--dmd2_discriminator_gan_loss_weight", type=float, default=0.01, help="Weight of real/fake GAN losses for fake model and classifier updates.")
    parser.add_argument("--dmd2_student_sigmas", type=str, required=True, help="Sigmas of the student model.")
    parser.add_argument("--dmd2_student_timesteps", type=str, required=True, help="Timesteps of the student model.")
    parser.add_argument("--dmd2_teacher_cfg", type=float, default=3., help="CFG scale of the teacher model.")
    parser.add_argument("--dmd2_noise_sampling", choices=["uniform", "beta"], default="uniform", help="DM and fake noise sampling: uniform or Beta(2.5, 1). GAN always uses uniform sampling.")
    parser.add_argument("--dmd2_min_timestep", type=int, default=20, help="Lower randint bound for uniform DM sampling; beta DM/fake sigma is clamped below at (value + 1) / 1000.")
    parser.add_argument("--dmd2_max_timestep", type=int, default=981, help="Exclusive upper randint bound for uniform DM sampling; beta DM/fake sigma is clamped above at value / 1000.")
    parser.add_argument("--dmd2_classifier_channels", type=int, default=1280)
    parser.add_argument("--dmd2_feature_module_of_dit_for_gan", type=str, help="Module within lora_base_model to hook for GAN features.")
    return parser

def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_quant_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_template_model_config(parser)
    parser = add_offload_training_config(parser)
    parser = add_logger_config(parser)
    return parser
