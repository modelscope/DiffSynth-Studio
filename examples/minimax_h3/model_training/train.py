import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadAudioWithTorchaudio, ToAbsolutePath
from diffsynth.utils.data.minimax_h3 import MiniMaxH3ReferenceLoader
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MINIMAX_H3_FRAME_RATE = 24
MINIMAX_H3_TIME_DIVISION_FACTOR = 17
MINIMAX_H3_TIME_DIVISION_REMAINDER = 5


class MiniMaxH3TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        template_model_id_or_path=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        silent_on_missing_audio=False,
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
        pipe_kwargs = {}
        if processor_path is not None:
            processor_config = self.parse_path_or_model_id(processor_path)
            pipe_kwargs["processor_config"] = ModelConfig(model_id=processor_config.model_id, origin_file_pattern=processor_config.origin_file_pattern)
        self.pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, **pipe_kwargs)
        self.pipe = self.load_training_template_model(self.pipe, template_model_id_or_path, use_gradient_checkpointing, use_gradient_checkpointing_offload)
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model,
            remove_unnecessary_params=True,
            force_remove_params_shared=("video_latents", "audio_latents"),
            force_remove_params_nega=("prompt_embeds", "text_token_tags", "packed"),
        )
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        self.pipe.scheduler_audio.set_timesteps(1000, training=True)

        # Store other configs
        self.silent_on_missing_audio = silent_on_missing_audio
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTMiniMaxH3AudioVideoLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTMiniMaxH3AudioVideoLoss(pipe, **inputs_shared, **inputs_posi),
        }

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        # First/last-frame conditioning is derived from the training video itself, following the
        # `input_image` / `end_image` convention used by the wanvideo series. The H3 pipeline takes
        # them as a `keyframes` list plus `keyframe_indices` in {0, -1}.
        keyframes, keyframe_indices = [], []
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                keyframes.append(data["video"][0])
                keyframe_indices.append(0)
            elif extra_input == "end_image":
                keyframes.append(data["video"][-1])
                keyframe_indices.append(-1)
            else:
                inputs_shared[extra_input] = data[extra_input]
        if keyframes:
            inputs_shared["keyframes"] = keyframes
            inputs_shared["keyframe_indices"] = keyframe_indices
        # If no audio tracks in the video file, we use a silence tensor for training.
        if self.silent_on_missing_audio:
            if "input_audio" in extra_inputs and "input_audio" in inputs_shared and inputs_shared["input_audio"] is None:
                inputs_shared["input_audio"] = (torch.zeros((2, 800 * round(len(data["video"]) / 24 * 40))), 32000)
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
            "keyframes": None,
            "keyframe_indices": None,
            "references": None,
            "ref_image_short_edge": 2048,
            "ref_video_short_edge": 768, "ref_video_max_pixels": 768 * 1344,
            "imgvid_cond_noise_aug": self.pipe.imgvid_cond_noise_aug,
            "audio_cond_noise_aug": self.pipe.audio_cond_noise_aug,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "seed": 42,
            "rand_device": "cpu",
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


def minimax_h3_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--processor_path", type=str, default=None, help="Path or `model_id:pattern` of the Qwen3-VL processor.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--silent_on_missing_audio", default=False, action="store_true", help="Whether to use silent audio as a fallback when no audio track is present in the video data.")
    return parser


if __name__ == "__main__":
    parser = minimax_h3_parser()
    args = parser.parse_args()
    if args.num_frames % MINIMAX_H3_TIME_DIVISION_FACTOR != MINIMAX_H3_TIME_DIVISION_REMAINDER:
        raise ValueError(
            f"--num_frames must be {MINIMAX_H3_TIME_DIVISION_FACTOR}n+{MINIMAX_H3_TIME_DIVISION_REMAINDER} "
            f"(e.g. 39, 56, 124) so it lands on the video VAE's temporal grouping, got {args.num_frames}."
        )
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
        time_division_factor=MINIMAX_H3_TIME_DIVISION_FACTOR,
        time_division_remainder=MINIMAX_H3_TIME_DIVISION_REMAINDER,
        frame_rate=MINIMAX_H3_FRAME_RATE,
        fix_frame_rate=True,
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=video_processor,
        special_operator_map={
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudioWithTorchaudio(
                num_frames=args.num_frames,
                time_division_factor=MINIMAX_H3_TIME_DIVISION_FACTOR,
                time_division_remainder=MINIMAX_H3_TIME_DIVISION_REMAINDER,
                frame_rate=MINIMAX_H3_FRAME_RATE,
                fix_frame_rate=True,
            ),
            "references": MiniMaxH3ReferenceLoader(
                base_path=args.dataset_base_path,
                height=args.height,
                width=args.width,
                max_pixels=args.max_pixels,
                num_frames=args.num_frames,
                frame_rate=MINIMAX_H3_FRAME_RATE,
            ),
        }
    )
    model = MiniMaxH3TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
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
        template_model_id_or_path=args.template_model_id_or_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        silent_on_missing_audio=args.silent_on_missing_audio,
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
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
