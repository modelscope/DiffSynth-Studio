"""
FLUX ControlNet LoRA Training with Per-Frame Consistency Loss.

Fine-tunes a LoRA on ControlNet using:
  L = L_flow_match(F1) + λ · ||v_θ(z_t, t, F1) - v_θ(z_t, t, F2)||²

where F1, F2 are different binary SPAD frames of the same scene.

Based on train_lora.py with the addition of:
  - PairedSPADDataset (provides two frames per sample)
  - Manual VAE encoding of F2 after pipeline units process F1
  - FlowMatchSFTWithConsistencyLoss
"""
import torch, os, argparse, accelerate, re
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.diffusion.consistency_loss import FlowMatchSFTWithConsistencyLoss
from diffsynth.utils.controlnet import ControlNetInput

from paired_spad_dataset import PairedSPADDataset
from train_lora import (
    FluxTrainingModule,
    flux_parser,
    convert_lora_format,
    log_sample_images,
    parse_resume_epoch,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FluxConsistencyTrainingModule(FluxTrainingModule):
    """Extends FluxTrainingModule with per-frame consistency loss."""

    def __init__(self, consistency_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.consistency_weight = consistency_weight
        self.task_to_loss["sft_consistency"] = self._consistency_loss
        self.task_to_loss["sft_consistency:train"] = self._consistency_loss

    def get_pipeline_inputs(self, data):
        f2_pil = data.pop("controlnet_image_f2", None)
        inputs_shared, inputs_posi, inputs_nega = super().get_pipeline_inputs(data)
        if f2_pil is not None:
            inputs_shared["_f2_pil"] = f2_pil
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if (not getattr(self.pipe.scheduler, "training", False)) or \
           (len(self.pipe.scheduler.timesteps) != self.pipe.scheduler.num_train_timesteps):
            self.pipe.scheduler.set_timesteps(self.pipe.scheduler.num_train_timesteps, training=True)

        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)

        f2_pil = inputs[0].pop("_f2_pil", None)

        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        if f2_pil is not None:
            self.pipe.load_models_to_device(["vae_encoder"])
            f2_tensor = self.pipe.preprocess_image(f2_pil).to(
                device=self.pipe.device, dtype=self.pipe.torch_dtype
            )
            f2_latent = self.pipe.vae_encoder(f2_tensor, tiled=False)
            inputs[0]["controlnet_conditionings_f2"] = [f2_latent]

        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss

    def _consistency_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        return FlowMatchSFTWithConsistencyLoss(
            pipe,
            consistency_weight=self.consistency_weight,
            **inputs_shared,
            **inputs_posi,
        )


def consistency_parser():
    parser = flux_parser()
    parser.add_argument("--consistency_weight", type=float, default=0.1,
                        help="Weight λ for consistency loss (default: 0.1)")
    parser.add_argument("--frame_folders", type=str, default=None,
                        help="Comma-separated list of frame folder names "
                             "(default: bits,bits_frame_1000,...,bits_frame_16000)")
    return parser


if __name__ == "__main__":
    parser = consistency_parser()
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=args.find_unused_parameters
        )],
    )

    frame_folders = None
    if args.frame_folders:
        from paired_spad_dataset import FRAME_FOLDERS as ALL_FRAMES
        folder_lookup = {f: t for f, t in ALL_FRAMES}
        frame_folders = [(f, folder_lookup[f]) for f in args.frame_folders.split(",")]

    print(f"[Dataset] Loading PairedSPADDataset from: {args.dataset_metadata_path}")
    dataset = PairedSPADDataset(
        base_path=args.dataset_base_path,
        metadata_csv=args.dataset_metadata_path,
        frame_folders=frame_folders,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
    )
    print(f"[Dataset] {len(dataset)} paired samples")

    val_dataloader = None
    if args.val_metadata_path is not None:
        print(f"[Val Dataset] Loading from: {args.val_metadata_path}")
        val_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.val_metadata_path,
            repeat=1,
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
        print(f"[Val Dataset] {len(val_dataset)} samples")
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False,
            collate_fn=lambda x: x[0],
            num_workers=args.dataset_num_workers,
        )

    # We are fine-tuning (warm-starting from a base checkpoint), NOT resuming
    # a previously interrupted training run. Always start from epoch 0.
    resume_epoch = None

    print(f"[Model] Initializing FLUX Consistency Training Module...")
    print(f"[Model] consistency_weight = {args.consistency_weight}")
    model = FluxConsistencyTrainingModule(
        consistency_weight=args.consistency_weight,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_1_path=getattr(args, 'tokenizer_1_path', None),
        tokenizer_2_path=getattr(args, 'tokenizer_2_path', None),
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
        task="sft_consistency",
        device=accelerator.device,
    )

    output_path_abs = os.path.abspath(args.output_path)
    print(f"[Checkpoint] Output directory: {output_path_abs}")
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=convert_lora_format if args.align_to_opensource_format else lambda x: x,
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[TensorBoard] Logging to: {log_dir}")

    from train_lora import launch_training_with_logging

    print(f"\n{'='*60}")
    print(f"FLUX ControlNet LoRA + Consistency Loss Training")
    print(f"{'='*60}")
    print(f"  Epochs: 0 → {args.num_epochs - 1}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Consistency weight: {args.consistency_weight}")
    print(f"  Frame folders: {len(dataset.frame_folders)}")
    print(f"  Checkpoints: {args.output_path}")
    print(f"{'='*60}\n")

    launch_training_with_logging(
        accelerator=accelerator,
        dataset=dataset,
        val_dataloader=val_dataloader,
        model=model,
        model_logger=model_logger,
        tb_writer=tb_writer,
        log_freq=args.log_freq,
        image_log_freq=args.image_log_freq,
        resume_epoch=resume_epoch,
        args=args,
    )

    tb_writer.close()
    print(f"\n✅ Consistency training complete! Checkpoints: {args.output_path}")
