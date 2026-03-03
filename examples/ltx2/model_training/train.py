import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadAudioWithTorchaudio, ToAbsolutePath, RouteByType, SequencialProcess
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LTX2TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
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
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized") if tokenizer_path is None else ModelConfig(tokenizer_path)
        self.pipe = LTX2AudioVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTAudioVideoLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTAudioVideoLoss(pipe, **inputs_shared, **inputs_posi),
        }
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "frame_rate": data.get("frame_rate", 24),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "video_patchifier": self.pipe.video_patchifier,
            "audio_patchifier": self.pipe.audio_patchifier,
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


def ltx2_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--frame_rate", type=float, default=24, help="frame rate of the training videos.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = ltx2_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    video_processor = UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=32,
            width_division_factor=32,
            num_frames=args.num_frames,
            time_division_factor=8,
            time_division_remainder=1,
        )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=video_processor,
        special_operator_map={
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudioWithTorchaudio(duration=float(args.num_frames) / float(args.frame_rate)),
            "in_context_videos": RouteByType(operator_map=[
                (str, video_processor),
                (list, SequencialProcess(video_processor)),
            ]),
        }
    )
    model = LTX2TrainingModule(
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
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
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
