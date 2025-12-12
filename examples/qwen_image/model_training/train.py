import torch, os, argparse, accelerate

# 触发 quanto / peft 的 workarounds
import diffsynth.utils.quantisation.quanto_workarounds  # noqa: F401
import diffsynth.utils.quantisation.peft_workarounds    # noqa: F401

from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.diffusion import *

from diffsynth.utils.quantisation import quantise_model, _quanto_model
from types import SimpleNamespace

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
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
        base_model_precision: str = "no_change",
        text_encoder_1_precision: str = "no_change",
        quantize_activations: bool = False,
        result_image_field_name: str = "result_image",
        quantize_vae: bool = False, # 可能会在非常细腻的纹理上有轻微劣化（通常可接受）
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        
        # 是否启用 quanto：只要 base_model_precision 或 text_encoder_1_precision 包含 "quanto"
        use_quanto = (
            (base_model_precision is not None and "quanto" in base_model_precision.lower())
            or (text_encoder_1_precision is not None and "quanto" in text_encoder_1_precision.lower())
        )

        load_device = "cpu" if use_quanto else device
        load_dtype = torch.bfloat16

        # 1. 先在 load_device 上加载整条 pipeline
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=load_dtype,
            device=load_device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            processor_config=processor_config,
        )

        # 2. 如果启用 quanto，对 DiT + 文本编码器做 SimpleTuner 风格量化
        if use_quanto:
            fake_args = SimpleNamespace(
                base_model_precision=base_model_precision,
                text_encoder_1_precision=text_encoder_1_precision,
                text_encoder_2_precision="no_change",
                text_encoder_3_precision="no_change",
                text_encoder_4_precision="no_change",
                quantize_activations=quantize_activations,
            )

            dit, text_encoders, _, _ = quantise_model(
                model=self.pipe.dit,
                text_encoders=[self.pipe.text_encoder],
                controlnet=None,
                ema=None,
                args=fake_args,
            )
            self.pipe.dit = dit
            if text_encoders is not None and len(text_encoders) > 0:
                self.pipe.text_encoder = text_encoders[0]
            
            # 2) 额外：量化 VAE
            if quantize_vae:
                if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
                    # 对 VAE 用和 base 模型同样的 precision / 配置
                    self.pipe.vae = _quanto_model(
                        model=self.pipe.vae,
                        model_precision=base_model_precision,      # 通常是 "int8-quanto"
                        base_model_precision=base_model_precision, # 用于 fallback 判定
                        quantize_activations=quantize_activations, # 建议 False，先只量化权重
                    )


        # 3. 把整个 pipeline 挪到 accelerator.device
        if load_device != device:
            self.pipe.to(device)
        self.pipe.device = device
        
        # 4. 保持原来的 split + peft LoRA 逻辑
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.result_image_field_name = result_image_field_name
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data[self.result_image_field_name],
            "height": data[self.result_image_field_name].size[1],
            "width": data[self.result_image_field_name].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor.")

    # 和 SimpleTuner 对齐的量化参数
    parser.add_argument(
        "--base_model_precision",
        type=str,
        default="no_change",
        choices=[
            "no_change",
            "fp32",
            "fp16",
            "bf16",
            "int2-quanto",
            "int4-quanto",
            "int8-quanto",
            "fp8-quanto",
            "fp8uz-quanto",
        ],
        help="Precision for DiT / main diffusion model. Use '*-quanto' to enable optimum.quanto.",
    )
    parser.add_argument(
        "--text_encoder_1_precision",
        type=str,
        default="no_change",
        help="Precision for the first text encoder. Defaults to no_change (i.e. bf16), like SimpleTuner configs.",
    )
    parser.add_argument(
        "--quantize_activations",
        action="store_true",
        help="When using quanto, also quantize activations in addition to weights.",
    )
    parser.add_argument(
        "--result_image_field_name",
        type=str,
        default="result_image",
        help="The field name of the image generated by the model in the dataset JSON.",
    )
    parser.add_argument(
        "--quantize_vae",
        action="store_true",
        help="Enabling quantized VAEs may result in slight degradation in very fine textures (generally acceptable).",
    )
    
    return parser


if __name__ == "__main__":
    parser = qwen_image_parser()
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
    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
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
        device=accelerator.device,
        base_model_precision=args.base_model_precision,
        text_encoder_1_precision=args.text_encoder_1_precision,
        quantize_activations=args.quantize_activations,
        result_image_field_name=args.result_image_field_name,
        quantize_vae=args.quantize_vae,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
