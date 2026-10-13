from examples.flux2.model_training.train import *
from peft.tuners.tuners_utils import BaseTunerLayer
from copy import deepcopy
import torch


def model_fn_teacher(self: BasePipeline, cfg_scale, inputs_shared, inputs_posi, inputs_nega, timestep):
    models = {"dit": self.dit_teacher}
    model_output = self.cfg_guided_model_fn(
        self.model_fn, cfg_scale,
        inputs_shared, inputs_posi, inputs_nega,
        **models, timestep=timestep,
    )
    return model_output


def model_fn_student(self: BasePipeline, cfg_scale, inputs_shared, inputs_posi, inputs_nega, timestep):
    models = {"dit": self.dit}
    model_output = self.cfg_guided_model_fn(
        self.model_fn, cfg_scale,
        inputs_shared, inputs_posi, inputs_nega,
        **models, timestep=timestep,
    )
    return model_output


def fetch_trajectory(self: BasePipeline, model_fn, inputs, sigmas, timesteps, cfg_scale, latents=None, progress_bar_cmd=lambda x: x, return_first_output=False):
    # Inputs
    # (Some computations are identical between the student and teacher, but their overhead is negligible. We skip optimizing them to keep the framework general.)
    inputs_shared, inputs_posi, inputs_nega = inputs
    inputs_shared["cfg_scale"] = cfg_scale
    for unit in self.units:
        inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
    if latents is not None:
        inputs_shared["latents"] = latents
    self.scheduler.sigmas, self.scheduler.timesteps = sigmas, timesteps

    # Denoising loop
    trajectory = []
    self.load_models_to_device(self.in_iteration_models)
    for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
        trajectory.append(inputs_shared["latents"])
        timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
        noise_pred = model_fn(self, cfg_scale, inputs_shared, inputs_posi, inputs_nega, timestep)
        if return_first_output:
            return noise_pred
        inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
    trajectory.append(inputs_shared["latents"])
    return trajectory


class OPDTrainingModule(Flux2ImageTrainingModule):
    def __init__(self, teacher_sigmas, teacher_timesteps, teacher_cfg, student_sigmas, student_timesteps, student_cfg, num_distill_steps=1, min_distill_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.teacher_sigmas = self.parse_float_list(teacher_sigmas)
        self.teacher_timesteps = self.parse_float_list(teacher_timesteps)
        self.teacher_cfg = teacher_cfg
        self.student_sigmas = self.parse_float_list(student_sigmas)
        self.student_timesteps = self.parse_float_list(student_timesteps)
        self.student_cfg = student_cfg
        self.num_distill_steps = num_distill_steps
        self.min_distill_idx = min_distill_idx
        # Copy DiT
        self.pipe.dit_teacher = deepcopy(self.pipe.dit)
        self.pipe.dit_teacher.requires_grad_(False)

    def parse_float_list(self, x):
        if isinstance(x, str):
            x = torch.tensor(list(map(float, x.split(","))))
        return x

    def align_sigmas(self, sigmas, timesteps, target_sigmas, target_timesteps, idx):
        sigma_start, sigma_end = target_sigmas[idx], target_sigmas[idx + 1] if idx + 1 < len(target_sigmas) else 0
        aligned_sigmas, aligned_timesteps = [sigma_start], [target_timesteps[idx]]
        for i in range(len(sigmas)):
            if sigmas[i] < sigma_start and sigmas[i] > sigma_end:
                aligned_sigmas.append(sigmas[i])
                aligned_timesteps.append(timesteps[i])
            elif sigmas[i] <= sigma_end:
                break
        return torch.tensor(aligned_sigmas), torch.tensor(aligned_timesteps)

    def switch_model(self, variant="student"):
        pass

    def forward(self, data, inputs=None):
        # Inputs
        inputs = self.get_pipeline_inputs(data)

        # Fetch student trajectory
        self.switch_model("student")
        with torch.no_grad():
            trajectory = fetch_trajectory(self.pipe, model_fn_student, deepcopy(inputs), self.student_sigmas, self.student_timesteps, self.student_cfg)

        # Resample and train
        loss = 0
        distill_idx = torch.randperm(len(self.student_sigmas) - self.min_distill_idx) + self.min_distill_idx
        for idx in distill_idx[:self.num_distill_steps].tolist():
            # Select an inference step
            idx = torch.randint(0, len(self.student_sigmas), (1,)).item()
            aligned_sigmas, aligned_timesteps = self.align_sigmas(self.teacher_sigmas, self.teacher_timesteps, self.student_sigmas, self.student_timesteps, idx)
            latents = trajectory[idx]

            # Fetch student output
            self.switch_model("student")
            student_output = fetch_trajectory(self.pipe, model_fn_student, deepcopy(inputs), aligned_sigmas, aligned_timesteps, self.student_cfg, latents=latents, return_first_output=True)

            # Fetch teacher output
            self.switch_model("teacher")
            with torch.no_grad():
                teacher_trajectory = fetch_trajectory(self.pipe, model_fn_teacher, deepcopy(inputs), aligned_sigmas, aligned_timesteps, self.teacher_cfg, latents=latents)
            teacher_output = self.pipe.scheduler.return_to_timestep(self.pipe.scheduler.timesteps[0], latents, teacher_trajectory[-1])
            
            # Loss
            loss = loss + torch.nn.functional.mse_loss(student_output.float(), teacher_output.float())
        return loss


def OPD_parser(parser):
    parser.add_argument("--opd_teacher_sigmas", type=str, default="", help="Sigmas of the teacher model.")
    parser.add_argument("--opd_teacher_timesteps", type=str, default="", help="Timesteps of the teacher model.")
    parser.add_argument("--opd_teacher_cfg", type=float, default=4.0, help="CFG scale of the teacher model.")
    parser.add_argument("--opd_student_sigmas", type=str, default="", help="Sigmas of the student model.")
    parser.add_argument("--opd_student_timesteps", type=str, default="", help="Timesteps of the student model.")
    parser.add_argument("--opd_student_cfg", type=float, default=1.0, help="CFG scale of the student model.")
    parser.add_argument("--opd_num_distill_steps", type=int, default=1, help="How many steps (in student trajectory) are selected for training.")
    parser.add_argument("--opd_min_distill_idx", type=int, default=0, help="How many high-noise steps to ignore (num_distill_steps <= len(student_sigmas) - min_distill_idx).")
    return parser


if __name__ == "__main__":
    parser = OPD_parser(flux2_parser())
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
        )
    )
    model = OPDTrainingModule(
        teacher_sigmas=args.opd_teacher_sigmas,
        teacher_timesteps=args.opd_teacher_timesteps,
        teacher_cfg=args.opd_teacher_cfg,
        student_sigmas=args.opd_student_sigmas,
        student_timesteps=args.opd_student_timesteps,
        student_cfg=args.opd_student_cfg,
        num_distill_steps=args.opd_num_distill_steps,
        min_distill_idx=args.opd_min_distill_idx,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        template_model_id_or_path=args.template_model_id_or_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_lora_hot_loading=args.enable_lora_hot_loading,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
    )
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
    launch_training_task(accelerator, dataset, model, model_logger, args=args)
