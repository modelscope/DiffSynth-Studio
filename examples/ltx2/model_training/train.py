import torch, os, argparse, accelerate
from PIL import Image
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadAudioWithTorchaudio, ToAbsolutePath, RouteByType, RouteByExtensionName, SequencialProcess, LoadImage, LoadVideo, LoadGIF, ToList
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.core.device.vram_guard import VRAMGuard
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def infer_reference_downscale_factor(target_height, target_width, ref_height, ref_width):
    # Reference videos train at an exact integer downscale of the target.
    # The ratio must be uniform across axes; anything else is a data error.
    if target_height == ref_height and target_width == ref_width:
        return 1
    if target_height % ref_height != 0 or target_width % ref_width != 0:
        raise ValueError(
            f"Target dimensions ({target_height}x{target_width}) must be exact multiples "
            f"of reference dimensions ({ref_height}x{ref_width})"
        )
    scale_h = target_height // ref_height
    scale_w = target_width // ref_width
    if scale_h != scale_w:
        raise ValueError(
            f"Reference scale must be uniform. Got height scale {scale_h} and width scale {scale_w}. "
            f"Target: {target_height}x{target_width}, Reference: {ref_height}x{ref_width}"
        )
    if scale_h < 1:
        raise ValueError(
            f"Reference dimensions ({ref_height}x{ref_width}) cannot be larger than "
            f"target dimensions ({target_height}x{target_width})"
        )
    return scale_h


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
        quant_options=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        first_frame_conditioning_p=1.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, quant_options=quant_options, device=device)
        tokenizer_config = ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized") if tokenizer_path is None else ModelConfig(tokenizer_path)
        self.pipe = LTX2AudioVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config)
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model,
            remove_unnecessary_params=True,
            force_remove_params_shared=("audio_latents", "video_latents"),
            force_remove_params_nega=("audio_context", "video_context")
        )
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
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
        self.first_frame_conditioning_p = first_frame_conditioning_p
        self.reference_downscale_factor = None
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTAudioVideoLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTAudioVideoLoss(pipe, **inputs_shared, **inputs_posi),
        }
        
    def note_reference_downscale_factor(self, factor):
        # All training samples must share one reference/target ratio. The first
        # batch fixes it; a later mismatch is a data error, not a new ratio.
        if self.reference_downscale_factor is None:
            self.reference_downscale_factor = int(factor)
        elif self.reference_downscale_factor != int(factor):
            raise ValueError(
                f"Inconsistent reference downscale factor across batches. "
                f"First batch had factor={self.reference_downscale_factor}, "
                f"but current batch has factor={int(factor)}. "
                f"All training samples must use the same reference/target resolution ratio."
            )
        return self.reference_downscale_factor

    def get_checkpoint_metadata(self):
        if self.reference_downscale_factor is not None:
            return {"reference_downscale_factor": self.reference_downscale_factor}
        return {}

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        input_images, input_images_indexes = [], []
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                # Per-sample draw: p=1.0 keeps pure I2V/I2AV; lower p mixes in
                # T2V samples so one adapter serves both. Drawn from the torch
                # RNG so runs reproduce under torch.manual_seed.
                if torch.rand(()).item() < self.first_frame_conditioning_p:
                    input_images.append(data["video"][0])
                    input_images_indexes.append(0)
            elif extra_input == "end_image":
                input_images.append(data["video"][-1])
                input_images_indexes.append(len(data["video"]) - 1)
            else:
                inputs_shared[extra_input] = data[extra_input]
        if len(input_images) > 0:
            inputs_shared["input_images"] = input_images
            inputs_shared["input_images_indexes"] = input_images_indexes
            inputs_shared["input_images_strength"] = 1.0
        return inputs_shared

    @staticmethod
    def normalize_in_context_value(value):
        # Processed reference_video is flat frames for one video, per-video
        # frames for several; the pipeline always wants a list of videos.
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], Image.Image):
            return [value]
        return value

    def apply_reference_video(self, inputs_shared):
        if "reference_video" in inputs_shared:
            ref_videos = self.normalize_in_context_value(inputs_shared.pop("reference_video"))
            if inputs_shared.get("in_context_videos") is None:
                inputs_shared["in_context_videos"] = ref_videos
            # Reference frames keep native resolution so the downscale ratio
            # is inferred from data, then validated for every reference video.
            target_h, target_w = inputs_shared["height"], inputs_shared["width"]
            for ref_video in ref_videos:
                ref_h, ref_w = ref_video[0].size[1], ref_video[0].size[0]
                self.note_reference_downscale_factor(
                    infer_reference_downscale_factor(target_h, target_w, ref_h, ref_w)
                )
            explicit = inputs_shared.get("in_context_downsample_factor")
            if explicit is None:
                inputs_shared["in_context_downsample_factor"] = self.reference_downscale_factor
            elif int(explicit) != self.reference_downscale_factor:
                raise ValueError(
                    f"Explicit in_context_downsample_factor={int(explicit)} disagrees with "
                    f"inferred reference ratio {self.reference_downscale_factor}."
                )
            inputs_shared["reference_downscale_factor"] = self.reference_downscale_factor
        elif inputs_shared.get("in_context_videos") is not None:
            # Legacy key: frames arrive pre-resized, so no ratio can be
            # inferred; record an explicit metadata factor when provided.
            explicit = inputs_shared.get("in_context_downsample_factor")
            if explicit is not None:
                self.note_reference_downscale_factor(int(explicit))
                inputs_shared["reference_downscale_factor"] = self.reference_downscale_factor
        return inputs_shared

    def get_pipeline_inputs(self, data):
        prompt = data.get("prompt", data.get("caption"))
        if prompt is None:
            raise KeyError("Training metadata must provide a 'prompt' or 'caption' key.")
        inputs_posi = {"prompt": prompt}
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
        inputs_shared = self.apply_reference_video(inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if isinstance(data, dict):
            # LTX-2.5 RoPE consumes physical coordinates in FP32. Casting a
            # split cache to BF16 loses information before model_fn can use it.
            position_keys = {
                "video_positions", "audio_positions", "ref_frames_positions",
                "in_context_video_positions",
            }
            return {
                key: self.transfer_data_to_device(
                    value, device, None if key in position_keys else torch_float_dtype
                )
                for key, value in data.items()
            }
        return super().transfer_data_to_device(data, device, torch_float_dtype)

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        else:
            # Split-training cache path: recover the inferred factor recorded
            # during data_process so checkpoint metadata survives the cache.
            shared = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            if isinstance(shared, dict) and shared.get("reference_downscale_factor") is not None:
                self.note_reference_downscale_factor(shared["reference_downscale_factor"])
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
    parser.add_argument("--first_frame_conditioning_p", type=float, default=1.0, help="Per-sample probability of first-frame conditioning when 'input_image' is in extra_inputs. 1.0 keeps pure I2V/I2AV; lower values mix T2V samples (IC-LoRA composable default 0.2).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--abort_vram_gb", type=float, default=None, help="On a GPU shared with a desktop: cap this process below this card-wide sysfs VRAM line and kill it if the line is crossed.")
    return parser


if __name__ == "__main__":
    parser = ltx2_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    # Armed before any weights load so the baseline is other users of the card.
    guard = VRAMGuard(args.abort_vram_gb) if args.abort_vram_gb is not None else None
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
            frame_rate=args.frame_rate,
            fix_frame_rate=True,
        )
    # Reference videos keep native resolution: the downscale ratio against the
    # target is inferred from data, then the pipeline resizes to the ratio.
    reference_video_processor = RouteByType(operator_map=[
        (str, ToAbsolutePath(args.dataset_base_path) >> RouteByExtensionName(operator_map=[
            (("jpg", "jpeg", "png", "webp", "bmp"), LoadImage() >> ToList()),
            (("gif",), LoadGIF(
                args.num_frames, 8, 1,
            )),
            (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                args.num_frames, 8, 1,
                frame_rate=args.frame_rate, fix_frame_rate=True,
            )),
        ])),
    ])
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=video_processor,
        special_operator_map={
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudioWithTorchaudio(num_frames=args.num_frames, time_division_factor=8, time_division_remainder=1, frame_rate=args.frame_rate),
            "in_context_videos": RouteByType(operator_map=[
                (str, video_processor),
                (list, SequencialProcess(video_processor)),
            ]),
            "reference_video": RouteByType(operator_map=[
                (str, reference_video_processor),
                (list, SequencialProcess(reference_video_processor)),
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
        quant_options=args.quant_options,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        first_frame_conditioning_p=args.first_frame_conditioning_p,
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
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
    if guard is not None:
        guard.report()
        guard.close()
