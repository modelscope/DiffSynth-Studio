from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from .runner import get_optimizer_class, initialize_deepspeed_gradient_checkpointing
from diffsynth.core import OffloadTrainingManager


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
    fake_score_learning_rate: Optional[float] = None
    discriminator_learning_rate: Optional[float] = None
    feature_indices: Optional[list[int]] = None
    teacher_cfg_scale: float = 1.0
    student_grad_clip_norm: Optional[float] = 10.0


def _get_dmd2_pipe_model(module, model_name):
    if model_name is None:
        return None
    return getattr(module.pipe, model_name)


def get_dmd2_student_model(module):
    return _get_dmd2_pipe_model(module, module.dmd2_student_model_name)


def get_dmd2_teacher_model(module):
    return _get_dmd2_pipe_model(module, module.dmd2_teacher_model_name)


def get_dmd2_fake_score_model(module):
    return _get_dmd2_pipe_model(module, module.dmd2_fake_score_model_name)


def get_dmd2_discriminator(module):
    return _get_dmd2_pipe_model(module, module.dmd2_discriminator_model_name)


def _dmd2_model_state_names(module, model_name, param_names=None):
    if model_name is None:
        return set()
    model = getattr(module.pipe, model_name, None)
    if model is None:
        return set()
    names = model.state_dict().keys() if param_names is None else param_names
    return {f"pipe.{model_name}.{name}" for name in names}


def export_dmd2_trainable_state_dict(module, state_dict, remove_prefix=None):
    student_names = _dmd2_model_state_names(module, module.dmd2_student_model_name, module._dmd2_student_param_names)
    state_names = set(student_names)
    if remove_prefix is None:
        state_names.update(_dmd2_model_state_names(module, module.dmd2_teacher_model_name))
        state_names.update(_dmd2_model_state_names(module, module.dmd2_fake_score_model_name))
        state_names.update(_dmd2_model_state_names(module, module.dmd2_discriminator_model_name))
    state_dict = {name: param for name, param in state_dict.items() if name in state_names}
    if remove_prefix is not None:
        state_dict = {name[len(remove_prefix):] if name.startswith(remove_prefix) else name: param for name, param in state_dict.items()}
    return state_dict


def set_dmd2_train_phase(module, student_phase: bool):
    student_model = get_dmd2_student_model(module)
    student_model.train(student_phase)
    for name, param in student_model.named_parameters():
        param.requires_grad = student_phase and name in module._dmd2_student_param_names

    fake_score_model = get_dmd2_fake_score_model(module)
    fake_score_model.train(not student_phase)
    for name, param in fake_score_model.named_parameters():
        param.requires_grad = (not student_phase) and name in module._dmd2_fake_score_param_names

    discriminator = get_dmd2_discriminator(module)
    if discriminator is not None:
        discriminator.train(not student_phase)
        discriminator.requires_grad_(not student_phase)


def _expand_like(value, target, dtype=None):
    if dtype is None:
        dtype = target.dtype
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=target.device, dtype=dtype)
    value = value.to(device=target.device, dtype=dtype)
    while value.ndim < target.ndim:
        value = value.view(*value.shape, 1)
    return value


def flow_to_x0(latents, flow, sigma):
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


def _gan_loss_generator(fake_logits):
    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    gan_loss = F.softplus(-fake_logits).mean()
    return gan_loss


def _gan_loss_discriminator(real_logits, fake_logits):
    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    assert real_logits.ndim == 2, f"real_logits has shape {real_logits.shape}"
    gan_loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
    return gan_loss


class DMD2Loss:
    def __init__(self, config: DMD2Config):
        self.config = config

    def _model_forward_x0(
        self,
        module,
        model,
        model_fn,
        latents,
        timestep,
        sigma,
        inputs_shared,
        inputs_posi,
    ):
        shared = dict(inputs_shared)
        posi = dict(inputs_posi)
        shared["latents"] = latents
        flow = model_fn(
            module.pipe,
            model,
            timestep=timestep,
            progress_id=0,
            num_inference_steps=1,
            inputs_shared=shared,
            inputs_posi=posi,
        )
        return flow_to_x0(latents, flow, sigma)

    def _model_forward_features(
        self,
        module,
        model,
        model_fn,
        latents,
        timestep,
        inputs_shared,
        inputs_posi,
    ):
        discriminator = get_dmd2_discriminator(module)
        if discriminator is None:
            raise ValueError("DMD2 feature extraction requires a discriminator.")
        if self.config.feature_indices is None:
            raise ValueError("DMD2 feature extraction requires `dmd2_feature_indices`.")
        shared = dict(inputs_shared)
        posi = dict(inputs_posi)
        shared["latents"] = latents
        return model_fn(
            module.pipe,
            model,
            timestep=timestep,
            progress_id=0,
            num_inference_steps=1,
            inputs_shared=shared,
            inputs_posi=posi,
            feature_indices=set(self.config.feature_indices),
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
            get_dmd2_teacher_model(module),
            module.dmd2_model_fn_teacher,
            latents,
            timestep,
            sigma,
            inputs_shared,
            inputs_posi,
        )
        if self.config.teacher_cfg_scale <= 1.0:
            return teacher_x0_pos


        x0_neg = self._model_forward_x0(
            module,
            get_dmd2_teacher_model(module),
            module.dmd2_model_fn_teacher,
            latents,
            timestep,
            sigma,
            inputs_shared,
            inputs_nega,
        )
        return x0_neg + float(self.config.teacher_cfg_scale) * (teacher_x0_pos - x0_neg)

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
            get_dmd2_student_model(module),
            module.dmd2_model_fn_student,
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
            get_dmd2_teacher_model(module),
            module.dmd2_model_fn_teacher,
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
                get_dmd2_fake_score_model(module),
                module.dmd2_model_fn_fake_score,
                perturbed_data,
                timestep,
                sigma,
                inputs_shared,
                inputs_posi,
            )

        with torch.no_grad():
            teacher_x0 = self._teacher_forward_x0(
                module, perturbed_data, timestep, sigma, inputs_shared, inputs_posi, inputs_nega
            )
        if self.config.gan_loss_weight > 0:
            fake_feat = self._model_forward_features(
                module,
                get_dmd2_teacher_model(module),
                module.dmd2_model_fn_teacher,
                perturbed_data,
                timestep,
                inputs_shared,
                inputs_posi,
            )
            fake_logits_gen = get_dmd2_discriminator(module)(fake_feat)
            gan_loss_gen = _gan_loss_generator(fake_logits_gen)
        else:
            gan_loss_gen = torch.zeros((), device=real_data.device, dtype=torch.float32)

        vsd_loss = _variational_score_distillation_loss(gen_data, teacher_x0.detach(), fake_score_x0)
        gan_loss_weighted = self.config.gan_loss_weight * gan_loss_gen
        loss = vsd_loss + gan_loss_weighted

        return {"total_loss": loss}

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
            get_dmd2_fake_score_model(module),
            module.dmd2_model_fn_fake_score,
            x_t_sg,
            timestep,
            sigma,
            inputs_shared,
            inputs_posi,
        )
        fake_score_loss = F.mse_loss(fake_score_x0.float(), gen_data.float(), reduction="mean")

        gan_loss_disc = torch.zeros_like(fake_score_loss)
        gan_loss_ar1 = torch.zeros_like(fake_score_loss)
        gan_loss_logit_reg = torch.zeros_like(fake_score_loss)
        if self.config.gan_loss_weight > 0:
            with torch.no_grad():
                fake_feat = self._model_forward_features(
                    module,
                    get_dmd2_teacher_model(module),
                    module.dmd2_model_fn_teacher,
                    x_t_sg,
                    timestep,
                    inputs_shared,
                    inputs_posi,
                )
                real_feat, real_timestep, real_sigma = self._compute_real_feat(
                    module, real_data, timestep, sigma, eps, inputs_shared, inputs_posi
                )
            discriminator = get_dmd2_discriminator(module)
            real_logits = discriminator(real_feat)
            fake_logits = discriminator(fake_feat)
            gan_loss_disc = _gan_loss_discriminator(real_logits, fake_logits)
            if self.config.gan_r1_reg_weight > 0:
                perturbed_real_alpha = real_data + self.config.gan_r1_reg_alpha * torch.randn_like(real_data)
                with torch.no_grad():
                    real_feat_alpha = self._model_forward_features(
                        module,
                        get_dmd2_teacher_model(module),
                        module.dmd2_model_fn_teacher,
                        perturbed_real_alpha,
                        real_timestep,
                        inputs_shared,
                        inputs_posi,
                    )
                real_logits_alpha = discriminator(real_feat_alpha)
                gan_loss_ar1 = F.mse_loss(real_logits, real_logits_alpha, reduction="mean")

        loss = (
            fake_score_loss
            + gan_loss_disc
            + self.config.gan_r1_reg_weight * gan_loss_ar1
        )
        return {"total_loss": loss}

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
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
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
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer

    optimizer_class = get_optimizer_class(customized_optimizer)
    config = model.dmd2_config
    fake_score_lr = config.fake_score_learning_rate or learning_rate
    discriminator_lr = config.discriminator_learning_rate or learning_rate

    student_model = get_dmd2_student_model(model)
    fake_score_model = get_dmd2_fake_score_model(model)
    discriminator = get_dmd2_discriminator(model)
    has_discriminator = discriminator is not None

    student_optimizer = optimizer_class(_trainable_params(student_model), lr=learning_rate, weight_decay=weight_decay)
    fake_score_optimizer = optimizer_class(fake_score_model.parameters(), lr=fake_score_lr, weight_decay=weight_decay)
    student_scheduler = torch.optim.lr_scheduler.ConstantLR(student_optimizer, factor=1.0, total_iters=1)
    fake_score_scheduler = torch.optim.lr_scheduler.ConstantLR(fake_score_optimizer, factor=1.0, total_iters=1)

    optimizers = {
        "student": student_optimizer,
        "fake_score": fake_score_optimizer,
        "student_scheduler": student_scheduler,
        "fake_score_scheduler": fake_score_scheduler,
    }
    prepare_items = [model, student_optimizer, fake_score_optimizer, student_scheduler, fake_score_scheduler]
    if has_discriminator:
        discriminator_optimizer = optimizer_class(discriminator.parameters(), lr=discriminator_lr, weight_decay=weight_decay)
        discriminator_scheduler = torch.optim.lr_scheduler.ConstantLR(discriminator_optimizer, factor=1.0, total_iters=1)
        optimizers["discriminator"] = discriminator_optimizer
        optimizers["discriminator_scheduler"] = discriminator_scheduler
        prepare_items.extend([discriminator_optimizer, discriminator_scheduler])

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    prepare_items.append(dataloader)

    if enable_model_cpu_offload:
        prepared = accelerator.prepare(*prepare_items[1:])
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(
            model,
            accelerator.device,
            enable_optimizer_cpu_offload,
            cpu_offload_split_threshold,
        )
        prepared_tail = list(prepared)
    else:
        model.to(device=accelerator.device)
        prepared = accelerator.prepare(*prepare_items)
        model = prepared[0]
        prepared_tail = list(prepared[1:])

    optimizers["student"] = prepared_tail.pop(0)
    optimizers["fake_score"] = prepared_tail.pop(0)
    optimizers["student_scheduler"] = prepared_tail.pop(0)
    optimizers["fake_score_scheduler"] = prepared_tail.pop(0)
    if has_discriminator:
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
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                if iteration % config.student_update_freq == 0 and config.student_grad_clip_norm is not None and config.student_grad_clip_norm > 0:
                    accelerator.clip_grad_norm_(_trainable_params(get_dmd2_student_model(accelerator.unwrap_model(model))), config.student_grad_clip_norm)
                for optimizer in current_optimizers:
                    optimizer.step()
                for scheduler in current_schedulers:
                    scheduler.step()
                for optimizer in current_optimizers:
                    optimizer.zero_grad(set_to_none=True)
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                iteration += 1
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)