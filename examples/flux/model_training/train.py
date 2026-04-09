import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FluxTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_1_path=None, tokenizer_2_path=None,
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
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_1_config = ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="tokenizer/") if tokenizer_1_path is None else ModelConfig(tokenizer_1_path)
        tokenizer_2_config = ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="tokenizer_2/") if tokenizer_2_path is None else ModelConfig(tokenizer_2_path)
        self.pipe = FluxImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_1_config=tokenizer_1_config, tokenizer_2_config=tokenizer_2_config)
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
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "embedded_guidance": 1,
            "t5_sequence_length": 512,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
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


def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_1_path", type=str, default=None, help="Path to CLIP tokenizer.")
    parser.add_argument("--tokenizer_2_path", type=str, default=None, help="Path to T5 tokenizer.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    return parser


def convert_lora_format(state_dict, alpha=None):
    prefix_rename_dict = {
        "single_blocks": "lora_unet_single_blocks",
        "blocks": "lora_unet_double_blocks",
    }
    middle_rename_dict = {
        "norm.linear": "modulation_lin",
        "to_qkv_mlp": "linear1",
        "proj_out": "linear2",
        "norm1_a.linear": "img_mod_lin",
        "norm1_b.linear": "txt_mod_lin",
        "attn.a_to_qkv": "img_attn_qkv",
        "attn.b_to_qkv": "txt_attn_qkv",
        "attn.a_to_out": "img_attn_proj",
        "attn.b_to_out": "txt_attn_proj",
        "ff_a.0": "img_mlp_0",
        "ff_a.2": "img_mlp_2",
        "ff_b.0": "txt_mlp_0",
        "ff_b.2": "txt_mlp_2",
    }
    suffix_rename_dict = {
        "lora_B.weight": "lora_up.weight",
        "lora_A.weight": "lora_down.weight",
    }
    state_dict_ = {}
    for name, param in state_dict.items():
        names = name.split(".")
        if names[-2] != "lora_A" and names[-2] != "lora_B":
            names.pop(-2)
        prefix = names[0]
        middle = ".".join(names[2:-2])
        suffix = ".".join(names[-2:])
        block_id = names[1]
        if middle not in middle_rename_dict:
            continue
        rename = prefix_rename_dict[prefix] + "_" + block_id + "_" + middle_rename_dict[middle] + "." + suffix_rename_dict[suffix]
        state_dict_[rename] = param
        if rename.endswith("lora_up.weight"):
            lora_alpha = alpha if alpha is not None else param.shape[-1]
            state_dict_[rename.replace("lora_up.weight", "alpha")] = torch.tensor((lora_alpha,))[0]
    return state_dict_


if __name__ == "__main__":
    parser = flux_parser()
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
    model = FluxTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_1_path=args.tokenizer_1_path,
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
        device=accelerator.device,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=convert_lora_format if args.align_to_opensource_format else lambda x:x,
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
