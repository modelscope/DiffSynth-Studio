import os, torch, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.diffusion import ModelLogger
from diffsynth.diffusion.dmd2_trainer import DMD2TrainingModule, DMD2_parser, launch_dmd2_training_task
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
from examples.qwen_image_21.model_training.train import qwen_image_21_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenImage21DMD2TrainingModule(DMD2TrainingModule):
    def __init__(
        self,
        student_sigmas, student_timesteps, teacher_cfg=1.,
        min_timestep_dmd=20, max_timestep_dmd=981,
        model_paths=None, model_id_with_origin_paths=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=128, lora_dtype=None, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gan=True, generator_gan_loss_weight=0.005, discriminator_gan_loss_weight=0.01,
        classifier_channels=1280, feature_module_of_dit_for_gan="transformer_blocks.16",
        use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
        extra_inputs=None, device="cpu",
    ):
        super().__init__()

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, device=device)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        self.pipe = QwenImage21Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, processor_config=processor_config)
        if use_gan and self.pipe.vae is None:
            raise ValueError("Qwen Image 2.1 GAN training requires VAE weights.")

        # Training mode
        self.switch_pipe_to_dmd2_training_mode(
            self.pipe,
            student_sigmas, student_timesteps, teacher_cfg,
            min_timestep_dmd, max_timestep_dmd,
            lora_base_model, lora_target_modules, lora_rank, lora_dtype, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            use_gan, generator_gan_loss_weight, discriminator_gan_loss_weight,
            classifier_channels, feature_module_of_dit_for_gan,
            trainable_models,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": " "}
        inputs_shared = {
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "use_flex_attention": False,
            "tiled": False,
            "tile_size": 256,
            "tile_stride": 192,
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def prepare_batch(self, data):
        inputs_shared, inputs_posi, inputs_nega = self.get_pipeline_inputs(data)
        inputs_shared["cfg_scale"] = self.teacher_cfg
        inputs_shared["use_kv_cache"] = False
        if not self.gan:
            inputs_shared["input_image"] = None
        inputs = self.transfer_data_to_device((inputs_shared, inputs_posi, inputs_nega), self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    @property
    def gan_feature_channels(self):
        return self.pipe.dit.inner_dim

    def reshape_gan_features(self, features, inputs_shared):
        batch, _, height, width = inputs_shared["noise"].shape
        # Text and reference images precede the target image tokens.
        return features[:, -height * width:].transpose(1, 2).reshape(batch, -1, height, width)


if __name__ == "__main__":
    parser = DMD2_parser(qwen_image_21_parser())
    parser.set_defaults(dmd2_teacher_cfg=1., feature_module_of_dit_for_gan="transformer_blocks.16")
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=32,
            width_division_factor=32,
            convert_RGB=False,
            convert_RGBA=True,
        ),
    )
    model = QwenImage21DMD2TrainingModule(
        student_sigmas=args.dmd2_student_sigmas,
        student_timesteps=args.dmd2_student_timesteps,
        teacher_cfg=args.dmd2_teacher_cfg,
        min_timestep_dmd=args.min_timestep_dmd,
        max_timestep_dmd=args.max_timestep_dmd,
        use_gan=args.gan,
        generator_gan_loss_weight=args.generator_gan_loss_weight,
        discriminator_gan_loss_weight=args.discriminator_gan_loss_weight,
        classifier_channels=args.classifier_channels,
        feature_module_of_dit_for_gan=args.feature_module_of_dit_for_gan,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_dtype=args.lora_dtype,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        device=accelerator.device,
    )
    model.resume_from_checkpoint(args.resume_from_checkpoint, args.remove_prefix_in_ckpt)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        enable_csv_log=args.enable_csv_log,
    )
    launch_dmd2_training_task(accelerator, dataset, model, model_logger, args=args)
