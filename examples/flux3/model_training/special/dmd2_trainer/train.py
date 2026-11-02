import os, torch, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.diffusion import ModelLogger
from diffsynth.diffusion.dmd2_trainer import DMD2TrainingModule, DMD2_parser, launch_dmd2_training_task
from diffsynth.pipelines.flux3_image import Flux3ImagePipeline, ModelConfig
from examples.flux3.model_training.train import flux3_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Flux3DMD2TrainingModule(DMD2TrainingModule):
    def __init__(
        self,
        student_sigmas, student_timesteps, teacher_cfg=3.5,
        min_timestep_dmd=20, max_timestep_dmd=981,
        model_paths=None, model_id_with_origin_paths=None, processor_path=None,
        lora_base_model="dit", lora_target_modules="", lora_rank=128, lora_dtype=None, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gan=True, generator_gan_loss_weight=0.005, discriminator_gan_loss_weight=0.01,
        classifier_channels=1280, feature_module_of_dit_for_gan="single_blocks.13",
        use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
        extra_inputs=None, device="cpu",
    ):
        super().__init__()

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, device=device)
        processor_config = self.parse_path_or_model_id(processor_path, default_value=ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/"))
        self.pipe = Flux3ImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, processor_config=processor_config)
        if self.pipe.dit.flow_map:
            raise ValueError("Euler DMD2 training requires flux-3-image-klein-base, not the two-time flow-map model.")
        if use_gan and self.pipe.vae is None:
            raise ValueError("FLUX.3 GAN training requires VAE weights.")

        # Training mode
        self.switch_pipe_to_dmd2_training_mode(
            self.pipe,
            student_sigmas, student_timesteps, teacher_cfg,
            min_timestep_dmd, max_timestep_dmd,
            lora_base_model, lora_target_modules, lora_rank, lora_dtype, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            use_gan, generator_gan_loss_weight, discriminator_gan_loss_weight,
            classifier_channels, feature_module_of_dit_for_gan,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def prepare_batch(self, data):
        inputs_shared, inputs_posi, inputs_nega = self.get_pipeline_inputs(data)
        inputs_shared["cfg_scale"] = self.teacher_cfg
        if not self.gan:
            inputs_shared["input_image"] = None
        inputs = self.transfer_data_to_device((inputs_shared, inputs_posi, inputs_nega), self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    def sample_noise(self, clean, distribution_matching, min_timestep_dmd=20, max_timestep_dmd=981):
        # Packed latents have no batch axis; all image tokens share one noise level.
        noisy, level, noise = super().sample_noise(clean.unsqueeze(0), distribution_matching, min_timestep_dmd, max_timestep_dmd)
        return noisy[0], level, noise[0]

    def distribution_matching_loss(self, student_sample, teacher, fake):
        return super().distribution_matching_loss(student_sample.unsqueeze(0), teacher.unsqueeze(0), fake.unsqueeze(0))

    @property
    def gan_feature_channels(self):
        return self.pipe.dit.emb_in["image"].out_features

    def reshape_gan_features(self, features, inputs_shared):
        height, width = inputs_shared["height"] // 16, inputs_shared["width"] // 16
        # The packed single-stream sequence starts with the target image tokens.
        return features[:height * width].transpose(0, 1).reshape(1, -1, height, width)


if __name__ == "__main__":
    parser = DMD2_parser(flux3_parser())
    parser.set_defaults(dmd2_teacher_cfg=3.5, feature_module_of_dit_for_gan="single_blocks.13")
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
            height_division_factor=16,
            width_division_factor=16,
        ),
    )
    model = Flux3DMD2TrainingModule(
        student_sigmas=args.dmd2_student_sigmas,
        student_timesteps=args.dmd2_student_timesteps,
        teacher_cfg=args.dmd2_teacher_cfg,
        min_timestep_dmd=args.min_timestep_dmd,
        max_timestep_dmd=args.max_timestep_dmd,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        processor_path=args.processor_path,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_dtype=args.lora_dtype,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gan=args.gan,
        generator_gan_loss_weight=args.generator_gan_loss_weight,
        discriminator_gan_loss_weight=args.discriminator_gan_loss_weight,
        classifier_channels=args.classifier_channels,
        feature_module_of_dit_for_gan=args.feature_module_of_dit_for_gan,
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
