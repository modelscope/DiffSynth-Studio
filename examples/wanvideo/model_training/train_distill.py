import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        teacher_model_paths=None, teacher_model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        distillation_weight=0.5,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)

        # Load teacher model
        teacher_model_configs = []
        if teacher_model_paths is not None:
            teacher_model_paths = json.loads(teacher_model_paths)
            teacher_model_configs += [ModelConfig(path=path) for path in teacher_model_paths]
        if teacher_model_id_with_origin_paths is not None:
            teacher_model_id_with_origin_paths = teacher_model_id_with_origin_paths.split(",")
            teacher_model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in teacher_model_id_with_origin_paths]
        self.teacher_pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda", model_configs=teacher_model_configs)
        self.teacher_pipe.eval()
        for p in self.teacher_pipe.parameters():
            p.requires_grad = False

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(self.pipe, lora_base_model, model)

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.distillation_weight = distillation_weight


    def forward_preprocess(self, data, pipe):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Pipeline units will automatically process the input parameters.
        for unit in pipe.units:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def forward(self, data):
        # Common noise and timestep
        max_timestep_boundary = int(self.max_timestep_boundary * self.pipe.scheduler.num_train_timesteps)
        min_timestep_boundary = int(self.min_timestep_boundary * self.pipe.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=torch.bfloat16, device=self.pipe.device)

        # Preprocess data for student
        student_inputs = self.forward_preprocess(data, self.pipe)
        noise = torch.randn_like(student_inputs['input_latents'])
        student_inputs["latents"] = self.pipe.scheduler.add_noise(student_inputs["input_latents"], noise, timestep)
        training_target = self.pipe.scheduler.training_target(student_inputs["input_latents"], noise, timestep)

        # Student prediction
        student_models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        student_pred = self.pipe.model_fn(**student_models, **student_inputs, timestep=timestep)

        # Student loss
        student_loss = torch.nn.functional.mse_loss(student_pred.float(), training_target.float())
        student_loss = student_loss * self.pipe.scheduler.training_weight(timestep)

        # Teacher prediction
        with torch.no_grad():
            teacher_inputs = self.forward_preprocess(data, self.teacher_pipe)
            teacher_inputs["latents"] = self.teacher_pipe.scheduler.add_noise(teacher_inputs["input_latents"].to(self.teacher_pipe.device), noise.to(self.teacher_pipe.device), timestep.to(self.teacher_pipe.device))
            teacher_models = {name: getattr(self.teacher_pipe, name) for name in self.teacher_pipe.in_iteration_models}
            teacher_pred = self.teacher_pipe.model_fn(**teacher_models, **teacher_inputs, timestep=timestep)

        # Distillation loss
        distillation_loss = torch.nn.functional.mse_loss(student_pred.float(), teacher_pred.float().to(student_pred.device))

        # Final loss
        loss = (1 - self.distillation_weight) * student_loss + self.distillation_weight * distillation_loss
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--teacher_model_paths", type=str, default=None, help="Paths to load teacher models. In JSON format.")
    parser.add_argument("--teacher_model_id_with_origin_paths", type=str, default=None, help="Teacher model ID with origin paths.")
    parser.add_argument("--distillation_weight", type=float, default=0.5, help="Weight for distillation loss.")
    args = parser.parse_args()
    dataset = VideoDataset(args=args)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        teacher_model_paths=args.teacher_model_paths,
        teacher_model_id_with_origin_paths=args.teacher_model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        distillation_weight=args.distillation_weight,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        find_unused_parameters=args.find_unused_parameters,
        num_workers=args.dataset_num_workers,
    )
