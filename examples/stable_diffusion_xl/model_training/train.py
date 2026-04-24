import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ToAbsolutePath, LoadImage, ImageCropAndResize, RouteByType, SequencialProcess
from diffsynth.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class StableDiffusionXLTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, tokenizer_2_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        # ===== 解析模型配置 =====
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        # ===== Tokenizer 配置 =====
        tokenizer_config = self.parse_path_or_model_id(tokenizer_path, default_value=ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer/"))
        tokenizer_2_config = self.parse_path_or_model_id(tokenizer_2_path, default_value=ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer_2/"))
        # ===== 构建 Pipeline =====
        self.pipe = StableDiffusionXLPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, tokenizer_2_config=tokenizer_2_config)
        # ===== 拆分 Pipeline Units =====
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # ===== 切换到训练模式 =====
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # ===== 其他配置 =====
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        # ===== 任务模式路由 =====
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
        }

    def get_pipeline_inputs(self, data):
        # ===== 正向提示词 =====
        inputs_posi = {"prompt": data["prompt"]}
        # ===== 负向提示词：训练不需要 =====
        inputs_nega = {"negative_prompt": ""}
        # ===== 共享参数 =====
        height = data["image"].size[1]
        width = data["image"].size[0]
        inputs_shared = {
            # ===== 核心字段映射 =====
            "input_image": data["image"],
            "height": height,
            "width": width,
            # ===== 框架控制参数 =====
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        # ===== SDXL 特有：add_time_ids (micro-conditioning) =====
        # 在 __call__ 中计算，但训练不跑 __call__，所以在这里注入
        text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim
        add_time_ids = [height, width, 0, 0, height, width]
        expected_add_embed_dim = self.pipe.unet.add_embedding.linear_1.in_features
        addition_time_embed_dim = self.pipe.unet.add_time_proj.num_channels
        passed_add_embed_dim = addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
                f"but a vector of {passed_add_embed_dim} was created."
            )
        inputs_posi["add_time_ids"] = torch.tensor([add_time_ids], dtype=self.pipe.torch_dtype, device=self.pipe.device)
        # ===== 额外字段注入 =====
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        # ===== 标准实现，不要修改 =====
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def stable_diffusion_xl_parser():
    parser = argparse.ArgumentParser(description="Stable Diffusion XL training.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--tokenizer_2_path", type=str, default=None, help="Path to tokenizer 2.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = stable_diffusion_xl_parser()
    args = parser.parse_args()
    # ===== Accelerator 配置 =====
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    # ===== 数据集定义 =====
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
            height_division_factor=8,
            width_division_factor=8,
        ),
        special_operator_map={
            "image": RouteByType(operator_map=[
                (str, ToAbsolutePath(args.dataset_base_path) >> LoadImage() >> ImageCropAndResize(args.height, args.width, args.max_pixels, 8, 8)),
                (list, SequencialProcess(ToAbsolutePath(args.dataset_base_path) >> LoadImage(convert_RGB=False, convert_RGBA=True) >> ImageCropAndResize(args.height, args.width, args.max_pixels, 8, 8))),
            ]),
        },
    )
    # ===== TrainingModule =====
    model = StableDiffusionXLTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        tokenizer_2_path=args.tokenizer_2_path,
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
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )
    # ===== ModelLogger =====
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    # ===== 任务路由 =====
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
