import copy
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from .runner import get_optimizer_class, initialize_deepspeed_gradient_checkpointing


def _parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(i) for i in value.split(",") if i != ""]


def _parse_float_list(value):
    if value is None or value == "":
        return None
    return [float(i) for i in value.split(",") if i != ""]


@dataclass
class DMD2Config:
    student_update_freq: int = 5
    student_sample_steps: int = 4
    student_sample_type: str = "sde"
    student_schedule: str = "uniform"
    student_t_list: Optional[list[float]] = None
    matching_t_min: float = 0.001
    matching_t_max: float = 0.999
    matching_t_sampling: str = "uniform"
    matching_t_mean: float = 0.0
    matching_t_std: float = 1.0
    gan_loss_weight: float = 0.03
    gan_r1_reg_weight: float = 0.0
    gan_r1_reg_alpha: float = 0.1
    gan_logit_reg_weight: float = 0.0
    fake_score_learning_rate: Optional[float] = None
    discriminator_learning_rate: Optional[float] = None
    feature_indices: Optional[list[int]] = None
    discriminator_hidden_dim: Optional[int] = None
    discriminator_num_blocks: Optional[int] = None
    teacher_cfg_scale: Optional[float] = None
    student_grad_clip_norm: Optional[float] = 10.0

    @classmethod
    def from_args(cls, args):
        return cls(
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
            gan_logit_reg_weight=args.dmd2_gan_logit_reg_weight,
            fake_score_learning_rate=args.dmd2_fake_score_learning_rate,
            discriminator_learning_rate=args.dmd2_discriminator_learning_rate,
            feature_indices=_parse_int_list(args.dmd2_feature_indices),
            discriminator_hidden_dim=args.dmd2_discriminator_hidden_dim,
            discriminator_num_blocks=args.dmd2_discriminator_num_blocks,
            teacher_cfg_scale=args.dmd2_teacher_cfg_scale,
            student_grad_clip_norm=args.dmd2_student_grad_clip_norm,
        )


def add_dmd2_config(parser):
    parser.add_argument("--dmd2_student_update_freq", type=int, default=5, help="Update student once every N DMD2 iterations.")
    parser.add_argument("--dmd2_student_sample_steps", type=int, default=4, help="Number of distilled student sampling steps.")
    parser.add_argument("--dmd2_student_sample_type", type=str, default="sde", choices=["sde", "ode"], help="Student sampling type used by the DMD2 objective.")
    parser.add_argument("--dmd2_student_schedule", type=str, default="uniform", choices=["uniform"], help="Student sigma schedule.")
    parser.add_argument("--dmd2_student_t_list", type=str, default=None, help="Optional student sigma schedule, including the final 0.")
    parser.add_argument("--dmd2_matching_t_min", type=float, default=0.001, help="Minimum matching sigma sampled for DMD2.")
    parser.add_argument("--dmd2_matching_t_max", type=float, default=0.999, help="Maximum matching sigma sampled for DMD2.")
    parser.add_argument("--dmd2_matching_t_sampling", type=str, default="uniform", choices=["uniform", "logitnormal"], help="Sample matching sigma.")
    parser.add_argument("--dmd2_matching_t_mean", type=float, default=0.0, help="Mean for logitnormal matching timestep sampling.")
    parser.add_argument("--dmd2_matching_t_std", type=float, default=1.0, help="Std for logitnormal matching timestep sampling.")
    parser.add_argument("--dmd2_gan_loss_weight", type=float, default=0.03, help="Generator GAN loss weight.")
    parser.add_argument("--dmd2_gan_r1_reg_weight", type=float, default=0.0, help="Approximate R1 regularization weight for the discriminator.")
    parser.add_argument("--dmd2_gan_r1_reg_alpha", type=float, default=0.1, help="Noise scale for approximate R1 regularization.")
    parser.add_argument("--dmd2_gan_logit_reg_weight", type=float, default=0.0, help="L2 regularization weight on discriminator logits.")
    parser.add_argument("--dmd2_fake_score_learning_rate", type=float, default=None, help="Learning rate for the fake score model.")
    parser.add_argument("--dmd2_discriminator_learning_rate", type=float, default=None, help="Learning rate for the discriminator.")
    parser.add_argument("--dmd2_feature_indices", type=str, default=None, help="DiT block indices used by the discriminator.")
    parser.add_argument("--dmd2_discriminator_hidden_dim", type=int, default=None, help="Hidden dimension of DiT features.")
    parser.add_argument("--dmd2_discriminator_num_blocks", type=int, default=None, help="Total Flux block count.")
    parser.add_argument("--dmd2_teacher_cfg_scale", type=float, default=None, help="CFG scale applied to the teacher x0 in DMD2.")
    parser.add_argument("--dmd2_student_grad_clip_norm", type=float, default=10.0, help="Clip student gradients to this norm.")
    return parser


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
    def __init__(self, feature_indices=None, num_blocks=None, inner_dim=None):
        super().__init__()
        if num_blocks is None:
            raise ValueError("`num_blocks` must be provided.")
        if inner_dim is None:
            raise ValueError("`inner_dim` must be provided.")
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
            raise ValueError(f"Expected list of {self.num_features} feature tensors, got {type(feats)} with length {len(feats) if isinstance(feats, list) else 'N/A'}.")
        logits = []
        for head, feat in zip(self.heads, feats):
            param = next(head.parameters())
            feat = feat.to(device=param.device, dtype=param.dtype)
            logits.append(head(feat))
        return torch.cat(logits, dim=1)


def _infer_dit_num_blocks(dit, default=40):
    if hasattr(dit, "blocks") and hasattr(dit, "single_blocks"):
        return len(dit.blocks) + len(dit.single_blocks)
    if hasattr(dit, "transformer_blocks") and hasattr(dit, "single_transformer_blocks"):
        return len(dit.transformer_blocks) + len(dit.single_transformer_blocks)
    return default


def _infer_dit_hidden_dim(dit, default=3072):
    return int(getattr(dit, "inner_dim", default))


def setup_dmd2_training(module, pipe, config: DMD2Config):
    module.dmd2_config = config
    module.teacher_dit = copy.deepcopy(pipe.dit).eval().requires_grad_(False)
    module.fake_score = copy.deepcopy(pipe.dit).train().requires_grad_(True)
    module.discriminator = None
    if config.gan_loss_weight > 0:
        discriminator_num_blocks = config.discriminator_num_blocks or _infer_dit_num_blocks(pipe.dit)
        discriminator_hidden_dim = config.discriminator_hidden_dim or _infer_dit_hidden_dim(pipe.dit)
        module.discriminator = FluxDMD2Discriminator(
            feature_indices=config.feature_indices,
            num_blocks=discriminator_num_blocks,
            inner_dim=discriminator_hidden_dim,
        )
    module.dmd2_loss = DMD2Loss(config)
    module._dmd2_student_param_names = {name for name, param in pipe.dit.named_parameters() if param.requires_grad}
    module._dmd2_fake_score_param_names = {name for name, _ in module.fake_score.named_parameters()}

    def input_processor(inputs_shared):
        return prepare_dmd2_pipeline_inputs(inputs_shared, config)

    def loss_fn(pipe, inputs_shared, inputs_posi, inputs_nega, iteration=None, **kwargs):
        return module.dmd2_loss(
            module,
            (inputs_shared, inputs_posi, inputs_nega),
            0 if iteration is None else iteration,
        )

    def state_dict_exporter(state_dict, remove_prefix=None):
        return export_dmd2_trainable_state_dict(module, state_dict, remove_prefix=remove_prefix)

    module.task_to_input_processor["dmd2"] = input_processor
    module.task_to_loss["dmd2"] = loss_fn
    module.task_to_state_dict_exporter["dmd2"] = state_dict_exporter


def prepare_dmd2_pipeline_inputs(inputs_shared, config: DMD2Config):
    if _cfg_enabled(config.teacher_cfg_scale):
        inputs_shared["cfg_scale"] = config.teacher_cfg_scale
    return inputs_shared


def export_dmd2_trainable_state_dict(module, state_dict, remove_prefix=None):
    student_names = {"pipe.dit." + name for name in module._dmd2_student_param_names}
    state_dict = {name: param for name, param in state_dict.items() if name in student_names}
    if remove_prefix is not None:
        state_dict = {name[len(remove_prefix):] if name.startswith(remove_prefix) else name: param for name, param in state_dict.items()}
    return state_dict


def set_dmd2_train_phase(module, student_phase: bool):
    module.pipe.dit.train(student_phase)
    for name, param in module.pipe.dit.named_parameters():
        param.requires_grad = student_phase and name in module._dmd2_student_param_names

    module.fake_score.train(not student_phase)
    for name, param in module.fake_score.named_parameters():
        param.requires_grad = (not student_phase) and name in module._dmd2_fake_score_param_names

    if module.discriminator is not None:
        module.discriminator.train(not student_phase)
        module.discriminator.requires_grad_(not student_phase)


def _expand_like(value, target, dtype=None):
    if dtype is None:
        dtype = target.dtype
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=target.device, dtype=dtype)
    value = value.to(device=target.device, dtype=dtype)
    while value.ndim < target.ndim:
        value = value.view(*value.shape, 1)
    return value


def _flow_to_x0(latents, flow, sigma):
    original_dtype = latents.dtype
    latents = latents.to(torch.float64)
    flow = flow.to(torch.float64)
    sigma = _expand_like(sigma, latents)
    return (latents - sigma * flow).to(original_dtype)


def _forward_process(x0, eps, sigma):
    original_dtype = x0.dtype
    x0 = x0.to(torch.float64)
    eps = eps.to(torch.float64)
    sigma = _expand_like(sigma, x0)
    return ((1 - sigma) * x0 + sigma * eps).to(original_dtype)


def _scale_noise(noise, sigma):
    original_dtype = noise.dtype
    noise = noise.to(torch.float64)
    sigma = _expand_like(sigma, noise)
    return (noise * sigma).to(original_dtype)


def make_dmd2_student_schedule(
    pipe,
    num_steps,
    device,
    student_schedule="uniform",
    student_t_list=None,
    matching_t_max=0.999,
):
    time_dtype = torch.float64
    if student_t_list is not None:
        sigmas = torch.tensor(student_t_list, device=device, dtype=time_dtype)
        if sigmas[-1].item() != 0:
            raise ValueError("`dmd2_student_t_list` must include a final 0.")
        if len(sigmas) != num_steps + 1:
            raise ValueError("The student sigma schedule length must equal `student_sample_steps + 1`.")
        timesteps = sigmas * pipe.scheduler.num_train_timesteps
        return sigmas, timesteps
    if student_schedule == "uniform":
        sigma_start = min(float(matching_t_max), 0.999)
        sigmas = torch.linspace(sigma_start, 0.0, num_steps + 1, device=device, dtype=time_dtype)
        timesteps = sigmas * pipe.scheduler.num_train_timesteps
        return sigmas, timesteps
    raise ValueError(f"Unsupported DMD2 student schedule: {student_schedule}")


def _variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0):
    dims = tuple(range(1, teacher_x0.ndim))
    with torch.no_grad():
        weight = 1 / ((gen_data.float() - teacher_x0.float()).abs().mean(dim=dims, keepdim=True) + 1e-6)
        weight = weight.to(dtype=gen_data.dtype)
        pseudo_target = gen_data - (fake_score_x0 - teacher_x0) * weight
    loss = 0.5 * F.mse_loss(gen_data.float(), pseudo_target.float(), reduction="mean")
    return loss


def _mean_abs_by_sample(value):
    dims = tuple(range(1, value.ndim))
    return value.detach().float().abs().mean(dim=dims).mean()


def _gan_loss_generator(fake_logits):
    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    gan_loss = F.softplus(-fake_logits).mean()
    return gan_loss


def _gan_loss_discriminator(real_logits, fake_logits):
    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    assert real_logits.ndim == 2, f"real_logits has shape {real_logits.shape}"
    gan_loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
    return gan_loss


def _cfg_enabled(cfg_scale):
    return cfg_scale is not None and abs(float(cfg_scale) - 1.0) > 1e-12


class DMD2Loss:
    def __init__(self, config: DMD2Config):
        self.config = config
        self._last_teacher_cfg_delta = None

    def _model_forward_x0(
        self,
        module,
        dit,
        latents,
        timestep,
        sigma,
        inputs_shared,
        inputs_posi,
    ):
        pipe = module.pipe
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        models["dit"] = dit
        shared = dict(inputs_shared)
        posi = dict(inputs_posi)
        shared["latents"] = latents
        flow = pipe.model_fn(
            **models,
            **shared,
            **posi,
            timestep=timestep,
            progress_id=0,
            num_inference_steps=1,
        )
        return _flow_to_x0(latents, flow, sigma)

    def _model_forward_features(
        self,
        module,
        dit,
        latents,
        timestep,
        inputs_shared,
        inputs_posi,
    ):
        if module.discriminator is None:
            raise ValueError("DMD2 feature extraction requires a discriminator.")
        pipe = module.pipe
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        models["dit"] = dit
        shared = dict(inputs_shared)
        posi = dict(inputs_posi)
        shared["latents"] = latents
        return pipe.model_fn(
            **models,
            **shared,
            **posi,
            timestep=timestep,
            progress_id=0,
            num_inference_steps=1,
            feature_indices=set(module.discriminator.feature_indices),
            return_features=True,
        )

    def _teacher_forward_x0(
        self,
        module,
        latents,
        timestep,
        sigma,
        inputs_shared,
        inputs_posi,
        inputs_nega,
    ):
        teacher_x0_pos = self._model_forward_x0(
            module,
            module.teacher_dit,
            latents,
            timestep,
            sigma,
            inputs_shared,
            inputs_posi,
        )
        if not _cfg_enabled(self.config.teacher_cfg_scale):
            self._last_teacher_cfg_delta = torch.zeros((), device=latents.device, dtype=torch.float32)
            return teacher_x0_pos

        x0_neg = self._model_forward_x0(
            module,
            module.teacher_dit,
            latents,
            timestep,
            sigma,
            inputs_shared,
            inputs_nega,
        )
        x0 = x0_neg + float(self.config.teacher_cfg_scale) * (teacher_x0_pos - x0_neg)
        self._last_teacher_cfg_delta = _mean_abs_by_sample(x0 - teacher_x0_pos)
        return x0

    def _teacher_forward_features(
        self,
        module,
        latents,
        timestep,
        inputs_shared,
        inputs_posi,
    ):
        return self._model_forward_features(
            module,
            module.teacher_dit,
            latents,
            timestep,
            inputs_shared,
            inputs_posi,
        )

    def _sample_matching_timestep(self, pipe, device, dtype, batch_size=1, inputs_shared=None, real_data=None):
        if self.config.matching_t_min > self.config.matching_t_max:
            raise ValueError("`dmd2_matching_t_min` must be <= `dmd2_matching_t_max`.")
        time_dtype = torch.float64
        if self.config.matching_t_sampling == "uniform":
            sigma = torch.rand(batch_size, device=device, dtype=time_dtype)
            sigma = sigma * (self.config.matching_t_max - self.config.matching_t_min) + self.config.matching_t_min
            timestep = sigma * pipe.scheduler.num_train_timesteps
            return timestep, sigma
        if self.config.matching_t_sampling == "logitnormal":
            sigma = torch.sigmoid(
                torch.randn(batch_size, device=device, dtype=time_dtype) * self.config.matching_t_std
                + self.config.matching_t_mean
            )
            sigma = sigma * (self.config.matching_t_max - self.config.matching_t_min) + self.config.matching_t_min
            sigma = sigma.clamp(self.config.matching_t_min, self.config.matching_t_max)
            timestep = sigma * pipe.scheduler.num_train_timesteps
            return timestep, sigma
        raise ValueError(f"Unsupported DMD2 matching timestep sampling: {self.config.matching_t_sampling}")

    def _generate_student_data(self, module, real_data, inputs_shared, inputs_posi):
        pipe = module.pipe
        device, dtype = real_data.device, real_data.dtype
        batch_size = real_data.shape[0]
        student_sigmas, student_timesteps = make_dmd2_student_schedule(
            pipe,
            self.config.student_sample_steps,
            device,
            student_schedule=self.config.student_schedule,
            student_t_list=self.config.student_t_list,
            matching_t_max=self.config.matching_t_max,
        )
        if len(student_sigmas) != self.config.student_sample_steps + 1:
            raise ValueError("The student sigma schedule length must equal `student_sample_steps + 1`.")

        if self.config.student_sample_steps == 1:
            timestep = student_timesteps[0:1].expand(batch_size)
            sigma = student_sigmas[0:1].expand(batch_size)
            input_student = _scale_noise(torch.randn_like(real_data), sigma)
        else:
            step_id = torch.randint(0, self.config.student_sample_steps, (batch_size,), device=device)
            sigma = student_sigmas[step_id]
            timestep = student_timesteps[step_id]
            eps_student = torch.randn_like(real_data)
            input_student = _forward_process(real_data, eps_student, sigma)
        gen_data = self._model_forward_x0(
            module,
            module.pipe.dit,
            input_student,
            timestep,
            sigma,
            inputs_shared,
            inputs_posi,
        )
        return gen_data, input_student

    def _compute_real_feat(self, module, real_data, timestep, sigma, eps, inputs_shared, inputs_posi):
        real_timestep, real_sigma, real_eps = timestep, sigma, eps
        perturbed_real = _forward_process(real_data, real_eps, real_sigma)
        real_feat = self._model_forward_features(
            module,
            module.teacher_dit,
            perturbed_real,
            real_timestep,
            inputs_shared,
            inputs_posi,
        )
        return real_feat, real_timestep, real_sigma

    def _student_update_step(self, module, real_data, inputs_shared, inputs_posi, inputs_nega):
        gen_data, input_student = self._generate_student_data(module, real_data, inputs_shared, inputs_posi)
        timestep, sigma = self._sample_matching_timestep(
            module.pipe,
            real_data.device,
            real_data.dtype,
            real_data.shape[0],
            inputs_shared=inputs_shared,
            real_data=real_data,
        )
        eps = torch.randn_like(real_data)
        perturbed_data = _forward_process(gen_data, eps, sigma)

        with torch.no_grad():
            fake_score_x0 = self._model_forward_x0(
                module,
                module.fake_score,
                perturbed_data,
                timestep,
                sigma,
                inputs_shared,
                inputs_posi,
            )

        if self.config.gan_loss_weight > 0:
            with torch.no_grad():
                teacher_x0 = self._teacher_forward_x0(
                    module, perturbed_data, timestep, sigma, inputs_shared, inputs_posi, inputs_nega
                )
            fake_feat = self._teacher_forward_features(
                module,
                perturbed_data,
                timestep,
                inputs_shared,
                inputs_posi,
            )
            fake_logits_gen = module.discriminator(fake_feat)
            gan_loss_gen = _gan_loss_generator(fake_logits_gen)
        else:
            with torch.no_grad():
                teacher_x0 = self._teacher_forward_x0(
                    module, perturbed_data, timestep, sigma, inputs_shared, inputs_posi, inputs_nega
                )
            gan_loss_gen = torch.zeros((), device=real_data.device, dtype=torch.float32)

        vsd_loss = _variational_score_distillation_loss(gen_data, teacher_x0.detach(), fake_score_x0)
        gan_loss_weighted = self.config.gan_loss_weight * gan_loss_gen
        loss = vsd_loss + gan_loss_weighted

        with torch.no_grad():
            teacher_delta = _mean_abs_by_sample(gen_data - teacher_x0)
            fake_delta = _mean_abs_by_sample(gen_data - fake_score_x0)
            vsd_delta = _mean_abs_by_sample(fake_score_x0 - teacher_x0)
            effective_gan_weight = gan_loss_weighted.detach() / gan_loss_gen.detach().clamp_min(1e-12)

        return {
            "total_loss": loss,
            "vsd_loss": vsd_loss.detach(),
            "gan_loss_gen": gan_loss_gen.detach(),
            "gan_loss_gen_weighted": gan_loss_weighted.detach(),
            "gan_loss_effective_weight": effective_gan_weight,
            "dmd2_teacher_delta": teacher_delta,
            "dmd2_teacher_cfg_delta": self._last_teacher_cfg_delta.detach(),
            "dmd2_fake_score_delta": fake_delta,
            "dmd2_vsd_delta": vsd_delta,
            "dmd2_sigma_mean": sigma.detach().float().mean(),
            "student_input_mean": input_student.detach().float().mean(),
        }

    def _fake_score_discriminator_update_step(self, module, real_data, inputs_shared, inputs_posi, inputs_nega):
        with torch.no_grad():
            gen_data, _ = self._generate_student_data(module, real_data, inputs_shared, inputs_posi)
            timestep, sigma = self._sample_matching_timestep(
                module.pipe,
                real_data.device,
                real_data.dtype,
                real_data.shape[0],
                inputs_shared=inputs_shared,
                real_data=real_data,
            )
            eps = torch.randn_like(real_data)
            x_t_sg = _forward_process(gen_data, eps, sigma)

        fake_score_x0 = self._model_forward_x0(
            module,
            module.fake_score,
            x_t_sg,
            timestep,
            sigma,
            inputs_shared,
            inputs_posi,
        )
        fake_score_loss = F.mse_loss(fake_score_x0.float(), gen_data.float(), reduction="mean")
        with torch.no_grad():
            fake_score_delta = _mean_abs_by_sample(fake_score_x0 - gen_data)

        gan_loss_disc = torch.zeros_like(fake_score_loss)
        gan_loss_ar1 = torch.zeros_like(fake_score_loss)
        gan_loss_logit_reg = torch.zeros_like(fake_score_loss)
        real_logit_mean = torch.zeros_like(fake_score_loss)
        fake_logit_mean = torch.zeros_like(fake_score_loss)
        if self.config.gan_loss_weight > 0:
            with torch.no_grad():
                fake_feat = self._model_forward_features(
                    module,
                    module.teacher_dit,
                    x_t_sg,
                    timestep,
                    inputs_shared,
                    inputs_posi,
                )
                real_feat, real_timestep, real_sigma = self._compute_real_feat(
                    module, real_data, timestep, sigma, eps, inputs_shared, inputs_posi
                )
            real_logits = module.discriminator(real_feat)
            fake_logits = module.discriminator(fake_feat)
            real_logit_mean = real_logits.detach().float().mean()
            fake_logit_mean = fake_logits.detach().float().mean()
            gan_loss_disc = _gan_loss_discriminator(real_logits, fake_logits)
            if self.config.gan_logit_reg_weight > 0:
                gan_loss_logit_reg = 0.5 * (real_logits.float().square().mean() + fake_logits.float().square().mean())
            if self.config.gan_r1_reg_weight > 0:
                perturbed_real_alpha = real_data + self.config.gan_r1_reg_alpha * torch.randn_like(real_data)
                with torch.no_grad():
                    real_feat_alpha = self._model_forward_features(
                        module,
                        module.teacher_dit,
                        perturbed_real_alpha,
                        real_timestep,
                        inputs_shared,
                        inputs_posi,
                    )
                real_logits_alpha = module.discriminator(real_feat_alpha)
                gan_loss_ar1 = F.mse_loss(real_logits, real_logits_alpha, reduction="mean")

        loss = (
            fake_score_loss
            + gan_loss_disc
            + self.config.gan_r1_reg_weight * gan_loss_ar1
            + self.config.gan_logit_reg_weight * gan_loss_logit_reg
        )
        return {
            "total_loss": loss,
            "fake_score_loss": fake_score_loss.detach(),
            "fake_score_delta": fake_score_delta,
            "gan_loss_disc": gan_loss_disc.detach(),
            "gan_loss_ar1": gan_loss_ar1.detach(),
            "gan_loss_logit_reg": gan_loss_logit_reg.detach(),
            "gan_loss_logit_reg_weighted": (self.config.gan_logit_reg_weight * gan_loss_logit_reg).detach(),
            "gan_real_logit": real_logit_mean,
            "gan_fake_logit": fake_logit_mean,
            "dmd2_sigma_mean": sigma.detach().float().mean(),
        }

    def __call__(self, module, inputs, iteration):
        inputs_shared, inputs_posi, inputs_nega = inputs
        real_data = inputs_shared.get("input_latents")
        if real_data is None:
            raise ValueError("DMD2 requires image latents from the dataset. Please provide training images.")
        student_phase = iteration % self.config.student_update_freq == 0
        set_dmd2_train_phase(module, student_phase)
        if student_phase:
            return self._student_update_step(module, real_data, inputs_shared, inputs_posi, inputs_nega)
        return self._fake_score_discriminator_update_step(module, real_data, inputs_shared, inputs_posi, inputs_nega)


def _trainable_params(module):
    return [param for param in module.parameters() if param.requires_grad]


def _dmd2_current_optimizers(config, optimizers, iteration):
    if iteration % config.student_update_freq == 0:
        return [optimizers["student"]], [optimizers["student_scheduler"]]
    current_optimizers = [optimizers["fake_score"]]
    current_schedulers = [optimizers["fake_score_scheduler"]]
    if "discriminator" in optimizers:
        current_optimizers.append(optimizers["discriminator"])
        current_schedulers.append(optimizers["discriminator_scheduler"])
    return current_optimizers, current_schedulers


def launch_dmd2_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model,
    model_logger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    customized_optimizer: str = None,
    args=None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        customized_optimizer = args.customized_optimizer

    optimizer_class = get_optimizer_class(customized_optimizer)
    config = model.dmd2_config
    fake_score_lr = config.fake_score_learning_rate or learning_rate
    discriminator_lr = config.discriminator_learning_rate or learning_rate

    student_optimizer = optimizer_class(_trainable_params(model.pipe.dit), lr=learning_rate, weight_decay=weight_decay)
    fake_score_optimizer = optimizer_class(model.fake_score.parameters(), lr=fake_score_lr, weight_decay=weight_decay)
    student_scheduler = torch.optim.lr_scheduler.ConstantLR(student_optimizer, factor=1.0, total_iters=1)
    fake_score_scheduler = torch.optim.lr_scheduler.ConstantLR(fake_score_optimizer, factor=1.0, total_iters=1)

    optimizers = {
        "student": student_optimizer,
        "fake_score": fake_score_optimizer,
        "student_scheduler": student_scheduler,
        "fake_score_scheduler": fake_score_scheduler,
    }
    prepare_items = [model, student_optimizer, fake_score_optimizer, student_scheduler, fake_score_scheduler]
    if model.discriminator is not None:
        discriminator_optimizer = optimizer_class(model.discriminator.parameters(), lr=discriminator_lr, weight_decay=weight_decay)
        discriminator_scheduler = torch.optim.lr_scheduler.ConstantLR(discriminator_optimizer, factor=1.0, total_iters=1)
        optimizers["discriminator"] = discriminator_optimizer
        optimizers["discriminator_scheduler"] = discriminator_scheduler
        prepare_items.extend([discriminator_optimizer, discriminator_scheduler])

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    prepare_items.append(dataloader)
    model.to(device=accelerator.device)
    prepared = accelerator.prepare(*prepare_items)

    model = prepared[0]
    prepared_tail = list(prepared[1:])
    optimizers["student"] = prepared_tail.pop(0)
    optimizers["fake_score"] = prepared_tail.pop(0)
    optimizers["student_scheduler"] = prepared_tail.pop(0)
    optimizers["fake_score_scheduler"] = prepared_tail.pop(0)
    if model.discriminator is not None:
        optimizers["discriminator"] = prepared_tail.pop(0)
        optimizers["discriminator_scheduler"] = prepared_tail.pop(0)
    dataloader = prepared_tail.pop(0)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    iteration = 0
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss_map = model({}, inputs=data, iteration=iteration)
                else:
                    loss_map = model(data, iteration=iteration)
                loss = loss_map["total_loss"]
                current_optimizers, current_schedulers = _dmd2_current_optimizers(config, optimizers, iteration)
                accelerator.backward(loss)
                if iteration % config.student_update_freq == 0 and config.student_grad_clip_norm is not None and config.student_grad_clip_norm > 0:
                    accelerator.clip_grad_norm_(_trainable_params(model.pipe.dit), config.student_grad_clip_norm)
                for optimizer in current_optimizers:
                    optimizer.step()
                for scheduler in current_schedulers:
                    scheduler.step()
                for optimizer in current_optimizers:
                    optimizer.zero_grad(set_to_none=True)
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss, metrics=loss_map)
                iteration += 1
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)