import torch, os, argparse, accelerate, random
from PIL import Image
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadAudioWithTorchaudio, ToAbsolutePath
from diffsynth.utils.data.minimax_h3 import MiniMaxH3ReferenceLoader
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MINIMAX_H3_FRAME_RATE = 24
MINIMAX_H3_TIME_DIVISION_FACTOR = 17
MINIMAX_H3_TIME_DIVISION_REMAINDER = 5


def get_random_mask(num_frames, height, width):
    """Draw one of the inpaint mask shapes the control branch was trained on.

    Returns a `(num_frames, height, width)` tensor of {0, 1}, where 1 marks the regions to
    regenerate. The shapes follow the reference training distribution: a whole clip, a centred
    block, a trailing segment, and a per-frame random block.
    """
    mask = torch.zeros((num_frames, height, width))
    shape = random.choices(["full", "block", "tail", "per_frame_block"], weights=[0.30, 0.35, 0.20, 0.15])[0]
    if shape == "full":
        mask[:] = 1
    elif shape == "block":
        block_h = random.randint(height // 4, height * 3 // 4)
        block_w = random.randint(width // 4, width * 3 // 4)
        top = random.randint(0, height - block_h)
        left = random.randint(0, width - block_w)
        mask[:, top:top + block_h, left:left + block_w] = 1
    elif shape == "tail":
        mask[random.randint(1, max(1, num_frames - 1)):] = 1
    else:
        for frame_id in range(num_frames):
            block_h = random.randint(1, max(1, height // 4))
            block_w = random.randint(1, max(1, width // 4))
            top = random.randint(0, height - block_h)
            left = random.randint(0, width - block_w)
            mask[frame_id, top:top + block_h, left:left + block_w] = 1
    return mask


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
        quant_options=None,
        template_model_id_or_path=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        silent_on_missing_audio=False,
        training_cfg_scale=1.0,
        control_dropout_prob=0.1,
        enable_inpaint=False,
        fully_masked_dropout_prob=0.9,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        if training_cfg_scale < 1.0:
            raise ValueError("training_cfg_scale must be at least 1.0")
        self.training_cfg_scale = training_cfg_scale
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, quant_options=quant_options, device=device)
        pipe_kwargs = {}
        if processor_path is not None:
            pipe_kwargs["processor_config"] = self.parse_path_or_model_id(processor_path)
        self.pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, **pipe_kwargs)
        self.pipe = self.load_training_template_model(self.pipe, template_model_id_or_path, use_gradient_checkpointing, use_gradient_checkpointing_offload)
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model,
            remove_unnecessary_params=True,
            force_remove_params_shared=("video_latents", "audio_latents"),
            force_remove_params_nega=("prompt_embeds", "text_token_tags", "packed") if training_cfg_scale == 1.0 else (),
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
        # Training the ControlNet injects its skips partway up the frozen main stack, so gradients
        # reach the branch only by flowing back through the DiT blocks above each injection point.
        # `freeze_except` leaves those blocks in eval mode, and DeepSpeed ZeRO-3 asserts
        # `sub_module.training` for every module it walks during backward. Keep the DiT in training
        # mode -- its parameters stay frozen through `requires_grad_(False)`, so nothing is updated;
        # only the mode flag changes, and MiniMax-H3 carries no dropout or batch-norm that would
        # alter the forward.
        if trainable_models is not None and "controlnet" in trainable_models.split(",") and self.pipe.dit is not None:
            self.pipe.dit.train()

        # Store other configs
        self.control_dropout_prob = control_dropout_prob
        self.enable_inpaint = enable_inpaint
        self.fully_masked_dropout_prob = fully_masked_dropout_prob
        self.silent_on_missing_audio = silent_on_missing_audio
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTMiniMaxH3AudioVideoLoss(
                pipe, training_cfg_scale=self.training_cfg_scale, inputs_nega=inputs_nega, **inputs_shared, **inputs_posi,
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTMiniMaxH3AudioVideoLoss(
                pipe, training_cfg_scale=self.training_cfg_scale, inputs_nega=inputs_nega, **inputs_shared, **inputs_posi,
            ),
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
        if "control_video" in extra_inputs:
            inputs_shared = self.parse_control_inputs(data, inputs_shared)
        return inputs_shared

    def parse_control_inputs(self, data, inputs_shared):
        # Two dropouts keep the released checkpoint's generality, mirroring the reference training.
        # Dropping the control rows keeps the unconditional path trainable and teaches the branch to
        # generate without a control video; a fully masked clip carries nothing of the original, so
        # most of those batches drop the inpaint condition entirely, which is the all-zero layout the
        # pipeline pads in for pure generation.
        if random.random() < self.control_dropout_prob:
            inputs_shared["control_video"] = None
        if self.enable_inpaint:
            frames = data["video"]
            width, height = frames[0].size
            mask = get_random_mask(len(frames), height, width)
            if bool(mask.min() == 1) and random.random() < self.fully_masked_dropout_prob:
                inputs_shared["inpaint_video"] = None
                inputs_shared["inpaint_video_mask"] = None
            else:
                inputs_shared["inpaint_video"] = frames
                inputs_shared["inpaint_video_mask"] = [Image.fromarray((m * 255).numpy().astype("uint8"), mode="L") for m in mask]
        if inputs_shared.get("control_video") is None and inputs_shared.get("inpaint_video_mask") is None:
            # Both conditions dropped: fall back to an all-zero control video so the branch still
            # receives its rows and every parameter keeps a gradient.
            inputs_shared["control_video"] = [Image.new("RGB", data["video"][0].size, (0, 0, 0))] * len(data["video"])
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": " "}
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
            "control_video": None, "control_scale": 1.0,
            "inpaint_video": None, "inpaint_video_mask": None,
            "tiled": True, "tile_size": 256, "tile_overlap": 64,
            "imgvid_cond_noise_aug": self.pipe.imgvid_cond_noise_aug,
            "audio_cond_noise_aug": self.pipe.audio_cond_noise_aug,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            # Reuse the pipeline's CFG preprocessing path to build unconditional
            # embeddings when CFG-aware training is enabled.
            "cfg_scale": self.training_cfg_scale,
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
    parser.add_argument("--training_cfg_scale", type=float, default=1.0, help="Inverse-CFG scale for preserving MiniMax-H3 guidance distillation during fine-tuning. Values greater than 1 enable a no-grad unconditional branch; 1 keeps the standard flow-matching loss.")
    parser.add_argument("--control_dropout_prob", type=float, default=0.1, help="Probability of dropping the control video of a batch, which keeps the unconditional path trainable.")
    parser.add_argument("--enable_inpaint", default=False, action="store_true", help="Whether to feed a random inpaint mask through the control branch alongside the control video. Requires a checkpoint whose control_in_dim covers the mask channels.")
    parser.add_argument("--fully_masked_dropout_prob", type=float, default=0.9, help="Probability of dropping the inpaint condition when the random mask covers the whole clip, so the all-zero layout reads as pure generation.")
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
        quant_options=args.quant_options,
        template_model_id_or_path=args.template_model_id_or_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        silent_on_missing_audio=args.silent_on_missing_audio,
        training_cfg_scale=args.training_cfg_scale,
        control_dropout_prob=args.control_dropout_prob,
        enable_inpaint=args.enable_inpaint,
        fully_masked_dropout_prob=args.fully_masked_dropout_prob,
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
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
