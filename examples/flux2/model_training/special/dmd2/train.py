import torch, os, argparse, accelerate
import copy
import math

from diffsynth.core import UnifiedDataset, gradient_checkpoint_forward
from diffsynth.diffusion import *
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig, model_fn_flux2

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(i) for i in value.split(",") if i != ""]


def _parse_float_list(value):
    if value is None or value == "":
        return None
    return [float(i) for i in value.split(",") if i != ""]


def _get_optimal_groups(num_channels):
    if num_channels <= 32:
        groups = max(1, num_channels // 4)
    else:
        groups = 32
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
    assert num_channels % groups == 0, f"{num_channels} not divisible by {groups}"
    return groups


class FluxDMD2Discriminator(torch.nn.Module):
    """DMD2 GAN discriminator scoring teacher hidden features at `feature_indices`.

    Only instantiated when `gan_loss_weight > 0`.
    """

    def __init__(self, feature_indices=None, num_blocks=40, inner_dim=3072):
        super().__init__()
        if feature_indices is None:
            feature_indices = [int(num_blocks // 2)]
        self.feature_indices = sorted({int(i) for i in feature_indices if 0 <= int(i) < num_blocks})
        if len(self.feature_indices) == 0:
            raise ValueError("DMD2 discriminator requires at least one valid feature index.")
        self.num_features = len(self.feature_indices)
        self.inner_dim = inner_dim

        hidden_channels = inner_dim // 2
        self.heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(inner_dim, hidden_channels, kernel_size=4, stride=2, padding=1),
                torch.nn.GroupNorm(_get_optimal_groups(hidden_channels), hidden_channels),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
            )
            for _ in self.feature_indices
        ])

    def forward(self, feats):
        if not isinstance(feats, list) or len(feats) != self.num_features:
            raise ValueError(
                f"Expected list of {self.num_features} feature tensors, "
                f"got {type(feats)} with length {len(feats) if isinstance(feats, list) else 'N/A'}."
            )
        logits = []
        for head, feat in zip(self.heads, feats):
            param = next(head.parameters())
            feat = feat.to(device=param.device, dtype=param.dtype)
            logits.append(head(feat))
        return torch.cat(logits, dim=1)

def model_fn_flux2_features(
    dit,
    latents=None,
    timestep=None,
    embedded_guidance=None,
    prompt_embeds=None,
    text_ids=None,
    image_ids=None,
    edit_latents=None,
    edit_image_ids=None,
    kv_cache=None,
    extra_text_embedding=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    feature_indices=None,
    **kwargs,
):
    """Flux.2 DiT forward exposing hidden features at `feature_indices` for the discriminator."""
    feature_indices = set() if feature_indices is None else set(feature_indices)
    image_seq_len = latents.shape[1]
    if edit_latents is not None:
        image_seq_len = latents.shape[1]
        latents = torch.concat([latents, edit_latents], dim=1)
        image_ids = torch.concat([image_ids, edit_image_ids], dim=1)
    if embedded_guidance is None:
        embedded_guidance = None
    elif isinstance(embedded_guidance, torch.Tensor):
        embedded_guidance = embedded_guidance.to(device=latents.device, dtype=latents.dtype).flatten()
        if embedded_guidance.numel() == 1:
            embedded_guidance = embedded_guidance.expand(latents.shape[0])
        elif embedded_guidance.numel() != latents.shape[0]:
            raise ValueError("`embedded_guidance` must be a scalar or match the latent batch size.")
    else:
        embedded_guidance = torch.full((latents.shape[0],), float(embedded_guidance), device=latents.device, dtype=latents.dtype)
    if extra_text_embedding is not None:
        extra_text_ids = torch.zeros((1, extra_text_embedding.shape[1], 4), dtype=text_ids.dtype, device=text_ids.device)
        extra_text_ids[:, :, -1] = torch.arange(prompt_embeds.shape[1], prompt_embeds.shape[1] + extra_text_embedding.shape[1])
        prompt_embeds = torch.concat([prompt_embeds, extra_text_embedding], dim=1)
        text_ids = torch.concat([text_ids, extra_text_ids], dim=1)

    height, width = kwargs.get("height"), kwargs.get("width")
    if height is not None and width is not None:
        feature_height, feature_width = int(height) // 16, int(width) // 16
    else:
        feature_height = int(math.sqrt(image_seq_len))
        feature_width = image_seq_len // feature_height if feature_height > 0 else 0
    if feature_height * feature_width != image_seq_len:
        raise ValueError("Flux2 feature extraction requires height/width or square latent tokens.")

    features = []

    def append_feature(feat):
        feat = feat[:, :image_seq_len]
        batch_size, _, channels = feat.shape
        feat = feat.permute(0, 2, 1).reshape(batch_size, channels, feature_height, feature_width)
        features.append(feat)
        if len(features) == len(feature_indices):
            return features
        return None

    num_txt_tokens = prompt_embeds.shape[1]
    timestep = timestep.to(latents.dtype)
    guidance = None if embedded_guidance is None else embedded_guidance.to(latents.dtype) * 1000
    temb = dit.time_guidance_embed(timestep, guidance)

    double_stream_mod_img = dit.double_stream_modulation_img(temb)
    double_stream_mod_txt = dit.double_stream_modulation_txt(temb)
    single_stream_mod = dit.single_stream_modulation(temb)[0]

    hidden_states = dit.x_embedder(latents)
    encoder_hidden_states = dit.context_embedder(prompt_embeds)

    if image_ids.ndim == 3:
        image_ids = image_ids[0]
    if text_ids.ndim == 3:
        text_ids = text_ids[0]

    image_rotary_emb = dit.pos_embed(image_ids)
    text_rotary_emb = dit.pos_embed(text_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    for block_id, block in enumerate(dit.transformer_blocks):
        encoder_hidden_states, hidden_states = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=double_stream_mod_img,
            temb_mod_params_txt=double_stream_mod_txt,
            image_rotary_emb=concat_rotary_emb,
            joint_attention_kwargs=None,
            kv_cache=None if kv_cache is None else kv_cache.get(f"double_{block_id}"),
        )
        if block_id in feature_indices:
            selected_features = append_feature(hidden_states)
            if selected_features is not None:
                return selected_features

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    num_double_blocks = len(dit.transformer_blocks)

    for block_id, block in enumerate(dit.single_transformer_blocks):
        hidden_states = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb_mod_params=single_stream_mod,
            image_rotary_emb=concat_rotary_emb,
            joint_attention_kwargs=None,
            kv_cache=None if kv_cache is None else kv_cache.get(f"single_{block_id}"),
        )
        feature_id = block_id + num_double_blocks
        if feature_id in feature_indices:
            selected_features = append_feature(hidden_states[:, num_txt_tokens:num_txt_tokens + image_seq_len])
            if selected_features is not None:
                return selected_features

    if len(features) != len(feature_indices):
        raise ValueError(f"Only collected {len(features)} feature maps for {len(feature_indices)} requested feature indices.")
    return features

def model_fn_flux2_dmd2(
    pipe,
    dit,
    timestep,
    progress_id,
    num_inference_steps,
    inputs_shared,
    inputs_posi,
    feature_indices=None,
    return_features=False,
):
    """Dispatcher used by `DMD2Loss`: returns flow prediction or hidden features."""
    if not return_features:
        return model_fn_flux2(
            dit=dit,
            **inputs_shared,
            **inputs_posi,
            timestep=timestep,
            progress_id=progress_id,
            num_inference_steps=num_inference_steps,
        )

    return model_fn_flux2_features(
        dit=dit,
        **inputs_shared,
        **inputs_posi,
        timestep=timestep,
        progress_id=progress_id,
        num_inference_steps=num_inference_steps,
        feature_indices=feature_indices,
    )


class Flux2DMD2TrainingModule(DiffusionTrainingModule):
    def __init__(self, args, device="cpu"):
        config = DMD2Config(
            student_update_freq=args.dmd2_student_update_freq,
            student_sample_steps=args.dmd2_student_sample_steps,
            student_sample_type=args.dmd2_student_sample_type,
            student_schedule=args.dmd2_student_schedule,
            student_t_list=_parse_float_list(args.dmd2_student_t_list),
            matching_t_min=args.dmd2_matching_t_min,
            matching_t_max=args.dmd2_matching_t_max,
            matching_t_sampling=args.dmd2_matching_t_sampling,
            matching_t_mean=args.dmd2_matching_t_mean,
            matching_t_std=args.dmd2_matching_t_std,
            gan_loss_weight=args.dmd2_gan_loss_weight,
            gan_r1_reg_weight=args.dmd2_gan_r1_reg_weight,
            gan_r1_reg_alpha=args.dmd2_gan_r1_reg_alpha,
            fake_score_learning_rate=args.dmd2_fake_score_learning_rate,
            discriminator_learning_rate=args.dmd2_discriminator_learning_rate,
            feature_indices=_parse_int_list(args.dmd2_feature_indices),
            teacher_cfg_scale=args.dmd2_teacher_cfg_scale,
            student_grad_clip_norm=args.dmd2_student_grad_clip_norm,
        )
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(args.model_paths, args.model_id_with_origin_paths, fp8_models=args.fp8_models, offload_models=args.offload_models, device=device)
        tokenizer_config = self.parse_path_or_model_id(args.tokenizer_path,default_value=ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="tokenizer/"))
        self.pipe = Flux2ImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device,model_configs=model_configs, tokenizer_config=tokenizer_config)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, args.trainable_models,
            args.lora_base_model, args.lora_target_modules, args.lora_rank, args.lora_checkpoint,
            args.preset_lora_path, args.preset_lora_model,
        )

        # DMD2 auxiliary models (frozen teacher, trainable fake_score, optional discriminator)
        self.pipe.dit_teacher = copy.deepcopy(self.pipe.dit)
        self.pipe.dit_fake_score = copy.deepcopy(self.pipe.dit)
        self.pipe.dmd2_discriminator = None
        if config.gan_loss_weight > 0:
            self.pipe.dmd2_discriminator = FluxDMD2Discriminator(feature_indices=config.feature_indices)
        self.pipe.dit_teacher.eval().requires_grad_(False)
        self.pipe.dit_fake_score.train().requires_grad_(True)
        if self.pipe.dmd2_discriminator is not None:
            self.pipe.dmd2_discriminator.train().requires_grad_(True)
        self.resume_from_checkpoint(args.resume_from_checkpoint, args.remove_prefix_in_ckpt)

        # Other configs
        self.use_gradient_checkpointing = args.use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = args.use_gradient_checkpointing_offload
        self.extra_inputs = args.extra_inputs.split(",") if args.extra_inputs is not None else []
        self.fp8_models = args.fp8_models
        self.embedded_guidance = args.embedded_guidance

        self.dmd2_student_model_name = "dit"
        self.dmd2_teacher_model_name = "dit_teacher"
        self.dmd2_fake_score_model_name = "dit_fake_score"
        self.dmd2_discriminator_model_name = "dmd2_discriminator"
        self.dmd2_model_fn_student = model_fn_flux2_dmd2
        self.dmd2_model_fn_teacher = model_fn_flux2_dmd2
        self.dmd2_model_fn_fake_score = model_fn_flux2_dmd2
        self.dmd2_config = config
        self.loss = DMD2Loss(config)
        self._dmd2_student_param_names = {name for name, param in self.pipe.dit.named_parameters() if param.requires_grad}
        self._dmd2_fake_score_param_names = {name for name, _ in self.pipe.dit_fake_score.named_parameters()}

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            "embedded_guidance": self.embedded_guidance,
            "cfg_scale": self.dmd2_config.teacher_cfg_scale,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None, iteration=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.loss(self, inputs, 0 if iteration is None else iteration)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        return export_dmd2_trainable_state_dict(self, state_dict, remove_prefix=remove_prefix)

def flux2_dmd2_parser():
    parser = argparse.ArgumentParser(description="Flux.2 DMD2 training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser = add_dmd2_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--embedded_guidance", type=float, default=1.0, help="Flux.2 embedded guidance value.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = flux2_dmd2_parser()
    args = parser.parse_args()
    args.find_unused_parameters = True

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
        ),
    )
    model = Flux2DMD2TrainingModule(
        args,
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
    launch_dmd2_training_task(accelerator, dataset, model, model_logger, args=args)
