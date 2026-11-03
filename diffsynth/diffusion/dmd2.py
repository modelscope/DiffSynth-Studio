import math, torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate.utils import broadcast, DistributedType
from diffsynth.core import load_state_dict
from .runner import exclude_quantized_params_from_ddp_sync, get_optimizer_class, save_training_args
from .training_module import DiffusionTrainingModule


class DMD2ClassifierHead(torch.nn.Module):
    def __init__(self, input_channels, channels=1280):
        super().__init__()
        groups = math.gcd(32, channels)
        if channels // groups < 2:
            raise ValueError("The classifier needs at least two channels per GroupNorm group.")
        layers = [torch.nn.AdaptiveAvgPool2d(32), torch.nn.Conv2d(input_channels, channels, 1)]
        for stride, padding in ((2, 1), (2, 1), (2, 1), (4, 0)):
            layers.extend([torch.nn.Conv2d(channels, channels, 4, stride, padding),
                           torch.nn.GroupNorm(groups, channels), torch.nn.SiLU()])
        self.layers = torch.nn.Sequential(*layers, torch.nn.Conv2d(channels, 1, 1))

    def forward(self, features):
        return self.layers(features.float()).flatten(1)


class DMD2FeatureExtractor:
    def __init__(self, model, module_path):
        self.module = model.get_submodule(module_path)

    def run(self, forward, *args, **kwargs):
        features = []
        handle = self.module.register_forward_hook(lambda module, inputs, output: features.append(output))
        try:
            forward(*args, **kwargs)
        finally:
            handle.remove()
        if len(features) != 1:
            raise ValueError("The GAN feature module must run exactly once per forward.")
        return features[0]


class DMD2TrainingModule(DiffusionTrainingModule):
    def switch_pipe_to_dmd2_training_mode(
        self, pipe,
        student_sigmas, student_timesteps, teacher_cfg=3.,
        min_timestep_dmd=20, max_timestep_dmd=981,
        lora_base_model=None, lora_target_modules="", lora_rank=128, lora_dtype=None, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gan=True, generator_gan_loss_weight=0.005, discriminator_gan_loss_weight=0.01,
        classifier_channels=1280, feature_module_of_dit_for_gan=None,
        trainable_models=None, noise_sampling="uniform",
    ):
        # Training config
        self.gan = use_gan
        self.teacher_cfg = teacher_cfg
        self.lora_base_model = lora_base_model
        self.trainable_models = trainable_models
        self.generator_gan_loss_weight = generator_gan_loss_weight
        self.discriminator_gan_loss_weight = discriminator_gan_loss_weight
        self.min_timestep_dmd = min_timestep_dmd
        self.max_timestep_dmd = max_timestep_dmd
        self.noise_sampling = noise_sampling
        if lora_base_model is None and trainable_models is None:
            raise ValueError("DMD2 requires trainable_models or lora_base_model.")

        # Freeze models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        if preset_lora_path is not None:
            pipe.load_lora(getattr(pipe, preset_lora_model), preset_lora_path)
        model = getattr(pipe, lora_base_model or trainable_models)
        self.adapter_layers = []

        if self.trainable_models is not None and self.lora_base_model is None:
            from copy import deepcopy
            self.fake_model = deepcopy(model)
            self.teacher_model = deepcopy(model).eval().requires_grad_(False)

        # Add LoRA
        if lora_base_model is not None:
            from peft.tuners.tuners_utils import BaseTunerLayer
            lora_dtype = pipe.torch_dtype if lora_dtype is None else {"bf16": torch.bfloat16, "fp32": torch.float32}[lora_dtype]
            model = self.add_lora_to_model(model, lora_target_modules, lora_rank, upcast_dtype=lora_dtype)
            model = self.add_lora_to_model(model, lora_target_modules, lora_rank, upcast_dtype=lora_dtype, adapter_name="fake")
            setattr(pipe, lora_base_model, model)

            # Load student LoRA
            if lora_checkpoint is not None:
                if isinstance(lora_checkpoint, str):
                    lora = load_state_dict(lora_checkpoint)
                else:
                    lora = lora_checkpoint
                lora_loader = pipe.lora_loader(torch_dtype=pipe.torch_dtype, device=pipe.device)
                lora = lora_loader.convert_state_dict(lora)
                lora = self.mapping_lora_state_dict(lora)
                load_result = model.load_state_dict(lora, strict=False)
                print(f"LoRA checkpoint loaded. Total {len(lora)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")

            self.adapter_layers = [layer for layer in model.modules() if isinstance(layer, BaseTunerLayer)]
            for layer in self.adapter_layers:
                layer.set_adapter(["default", "fake"])

        # Initialize GAN
        self.classifier = self.feature_extractor = None
        if self.gan:
            if feature_module_of_dit_for_gan is None:
                raise ValueError("GAN training requires feature_module_of_dit_for_gan.")
            self.classifier = DMD2ClassifierHead(self.gan_feature_channels, classifier_channels).to(pipe.device)
            self.feature_extractor = DMD2FeatureExtractor(self.fake_model if self.trainable_models is not None and self.lora_base_model is None else model, feature_module_of_dit_for_gan)

        # Set student schedule
        pipe.scheduler.sigmas = self.parse_float_list(student_sigmas)
        pipe.scheduler.timesteps = self.parse_float_list(student_timesteps)
        pipe.scheduler.training = True

    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None, adapter_name="default"):
        from peft import LoraConfig, inject_adapter_in_model
        if lora_alpha is None:
            lora_alpha = lora_rank
        target_modules = self.parse_lora_target_modules(model, target_modules)
        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model, adapter_name=adapter_name)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model

    def parse_float_list(self, x):
        return torch.tensor(list(map(float, x.split(",")))) if isinstance(x, str) else x

    def select_model(self, model_role):
        if self.trainable_models is not None and self.lora_base_model is None:
            return {"generator": getattr(self.pipe, self.trainable_models), "fake": self.fake_model, "teacher": self.teacher_model}[model_role]
        adapter = {"generator": "default", "fake": "fake", "teacher": []}[model_role]
        for layer in self.adapter_layers:
            layer.set_adapter(adapter)
        return getattr(self.pipe, self.lora_base_model)

    def generator_parameters(self):
        if self.trainable_models is not None and self.lora_base_model is None:
            return list(getattr(self.pipe, self.trainable_models).parameters())
        model = getattr(self.pipe, self.lora_base_model)
        return [p for name, p in model.named_parameters() if ".lora_A.default." in name or ".lora_B.default." in name]

    def critic_parameters(self):
        if self.trainable_models is not None and self.lora_base_model is None:
            parameters = list(self.fake_model.parameters())
        else:
            model = getattr(self.pipe, self.lora_base_model)
            parameters = [p for name, p in model.named_parameters() if ".lora_A.fake." in name or ".lora_B.fake." in name]
        return parameters + (list(self.classifier.parameters()) if self.gan else [])

    def prepare_batch(self, data):
        """Subclasses must return (inputs_shared, inputs_posi, inputs_nega) for their pipeline."""
        raise NotImplementedError("Implement prepare_batch in the model's training module.")

    def predict(self, x, timestep, inputs, model_role, features=False):
        model = self.select_model(model_role)
        inputs_shared, inputs_posi, inputs_nega = inputs
        inputs_shared = dict(inputs_shared, latents=x.to(self.pipe.torch_dtype))
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        models[self.lora_base_model or self.trainable_models] = model
        if features:
            output = self.feature_extractor.run(self.pipe.model_fn, **models, **inputs_shared, **inputs_posi, timestep=timestep)
            output = self.reshape_gan_features(output, inputs_shared)
        else:
            output = self.pipe.cfg_guided_model_fn(
                lambda **kwargs: self.pipe.model_fn(**kwargs).float(),
                self.teacher_cfg if model_role == "teacher" else 1.,
                inputs_shared, inputs_posi, inputs_nega, **models, timestep=timestep,
            )
        return output.float()

    def rollout(self, inputs, step, track_grad=False):
        x = inputs[0]["noise"].float()
        for index in range(step + 1):
            with torch.set_grad_enabled(track_grad and index == step):
                x = x.to(self.pipe.torch_dtype).float()
                timestep = self.pipe.scheduler.timesteps[index].to(x.device).reshape(1)
                velocity = self.predict(x, timestep, inputs, "generator")
                x = self.pipe.scheduler.step(velocity, timestep, x, to_final=index == step)
        return x

    def sample_noise(self, clean, distribution_matching, min_timestep_dmd=20, max_timestep_dmd=981, noise_sampling=None):
        noise_sampling = self.noise_sampling if noise_sampling is None else noise_sampling
        if noise_sampling == "beta":
            beta = torch.distributions.Beta(torch.tensor(2.5, device=clean.device), torch.tensor(1., device=clean.device))
            level = beta.sample((clean.shape[0],)).clamp((min_timestep_dmd + 1) / 1000, max_timestep_dmd / 1000)
        else:
            low, high = (min_timestep_dmd, max_timestep_dmd) if distribution_matching else (0, 1000)
            level = (torch.randint(low, high, (clean.shape[0],), device=clean.device).float() + 1) / 1000
        noise = torch.randn_like(clean)
        sigma = level.reshape(-1, *([1] * (clean.ndim - 1)))
        return (1 - sigma) * clean + sigma * noise, level, noise

    def teacher_clean(self, noised, inputs):
        noisy, level, _ = noised
        velocity = self.predict(noisy, level * 1000, inputs, "teacher")
        return noisy - level.reshape(-1, *([1] * (noisy.ndim - 1))) * velocity

    def fake_clean(self, noised, inputs):
        noisy, level, _ = noised
        velocity = self.predict(noisy, level * 1000, inputs, "fake")
        return noisy - level.reshape(-1, *([1] * (noisy.ndim - 1))) * velocity

    def logits(self, clean, inputs):
        noisy, level, _ = self.sample_noise(clean, False, noise_sampling="uniform")
        features = self.predict(noisy, level * 1000, inputs, "fake", features=True)
        return self.classifier(features)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        generator_param_ids = {id(param) for param in self.generator_parameters()}
        trainable_param_names = {name for name, param in self.named_parameters() if id(param) in generator_param_ids}
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict = {name.removeprefix(remove_prefix): param for name, param in state_dict.items()}
        return state_dict

    def distribution_matching_loss(self, student_sample, teacher, fake):
        with torch.no_grad():
            dims = tuple(range(1, student_sample.ndim))
            scale = (student_sample.float() - teacher.float()).abs().mean(dim=dims, keepdim=True).clamp_min(1e-8)
            direction = torch.nan_to_num((fake.float() - teacher.float()) / scale)
        return 0.5 * F.mse_loss(student_sample.float(), (student_sample.float() - direction).detach())

    def forward_generator(self, inputs, rollout_step, update_generator):
        if update_generator:
            student_sample = self.rollout(inputs, rollout_step, True)
            with torch.no_grad():
                noised = self.sample_noise(student_sample.detach(), True, self.min_timestep_dmd, self.max_timestep_dmd)
                teacher = self.teacher_clean(noised, inputs)
                fake = self.fake_clean(noised, inputs)
            loss = self.distribution_matching_loss(student_sample, teacher, fake)
            if self.gan:
                gan_sample = student_sample.detach().requires_grad_(True)
                logits = self.logits(gan_sample, inputs)
                gan_loss = F.softplus(-logits).mean()
                gradient = torch.autograd.grad(gan_loss, gan_sample)[0].detach()
                loss = loss + self.generator_gan_loss_weight * (gan_loss.detach() + ((student_sample - student_sample.detach()) * gradient).sum())
            self.select_model("generator")
        else:
            with torch.no_grad():
                student_sample = self.rollout(inputs, rollout_step)
            loss = None
        return loss, student_sample.detach()

    def forward_critic(self, inputs, student_sample):
        inputs_shared, inputs_posi, inputs_nega = inputs
        noisy, level, noise = self.sample_noise(student_sample.detach(), False, self.min_timestep_dmd, self.max_timestep_dmd)
        velocity = self.predict(noisy, level * 1000, inputs, "fake")
        loss = F.mse_loss(velocity.float(), noise.float() - student_sample.detach().float())
        if self.gan:
            real_logits = self.logits(inputs_shared['input_latents'].float(), inputs)
            fake_logits = self.logits(student_sample.detach(), inputs)
            loss = loss + self.discriminator_gan_loss_weight * (F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean())
        return loss

    def forward(self, data, phase, rollout_step=None, update_generator=True, inputs=None, student_sample=None):
        if phase == 'generator':
            with torch.no_grad():
                inputs = self.prepare_batch(data)
            loss, student_sample = self.forward_generator(inputs, rollout_step, update_generator)
            return loss, inputs, student_sample
        elif phase == 'critic':
            return self.forward_critic(inputs, student_sample)
        else:
            raise ValueError(f"Unknown DMD2 phase: {phase}")


def launch_dmd2_training_task(
    accelerator, dataset, model, model_logger,
    learning_rate=1e-4, weight_decay=0.01, num_workers=0, save_steps=None, num_epochs=1,
    enable_model_cpu_offload=False, customized_optimizer=None,
    fake_updates_per_generator=5, fake_warmup_steps=100, fake_learning_rate=None,
    args=None, **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        customized_optimizer = args.customized_optimizer
        fake_updates_per_generator = args.dmd2_fake_updates_per_generator
        fake_warmup_steps = args.dmd2_fake_warmup_steps
        fake_learning_rate = args.dmd2_fake_learning_rate
    num_steps = len(model.pipe.scheduler.sigmas)

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        raise ValueError("DMD2 training does not support DeepSpeed.")
    if enable_model_cpu_offload:
        raise ValueError("DMD2 training does not support model CPU offload.")
    if fake_updates_per_generator < 1:
        raise ValueError('Update ratio must be positive.')
    if fake_warmup_steps < 0:
        raise ValueError('Fake warmup steps must be nonnegative.')
    if accelerator.is_main_process:
        save_training_args(args)

    generator_params, critic_params = model.generator_parameters(), model.critic_parameters()
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda batch: batch[0], num_workers=num_workers)
    optimizer_class = get_optimizer_class(customized_optimizer)
    fake_learning_rate = learning_rate if fake_learning_rate is None else fake_learning_rate
    generator_optimizer = optimizer_class(generator_params, lr=learning_rate, weight_decay=weight_decay)
    critic_optimizer = optimizer_class(critic_params, lr=fake_learning_rate, weight_decay=weight_decay)
    generator_scheduler = torch.optim.lr_scheduler.ConstantLR(generator_optimizer, factor=1.0)
    critic_scheduler = torch.optim.lr_scheduler.ConstantLR(critic_optimizer, factor=1.0)

    model.to(device=accelerator.device)
    exclude_quantized_params_from_ddp_sync(accelerator, model)
    model, generator_optimizer, critic_optimizer, dataloader, generator_scheduler, critic_scheduler = accelerator.prepare(
        model, generator_optimizer, critic_optimizer, dataloader, generator_scheduler, critic_scheduler,
    )

    for epoch in range(num_epochs):
        for data in (progress_bar := tqdm(dataloader, disable=not accelerator.is_main_process)):
            if accelerator.sync_gradients:
                rollout_step = broadcast(torch.randint(num_steps, (), device=accelerator.device)).item()
                step = model_logger.num_steps - fake_warmup_steps
                update_generator = step >= 0 and step % fake_updates_per_generator == 0
            with accelerator.accumulate(model):
                with torch.set_grad_enabled(update_generator):
                    loss, inputs, student_sample = model(data, phase='generator', rollout_step=rollout_step, update_generator=update_generator)
                if update_generator:
                    accelerator.backward(loss)
                loss = model({}, phase='critic', inputs=inputs, student_sample=student_sample)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if update_generator:
                        accelerator.clip_grad_norm_(generator_params, 10.)
                    accelerator.clip_grad_norm_(critic_params, 10.)

                if update_generator:
                    generator_optimizer.step()
                    generator_scheduler.step()
                critic_optimizer.step()
                critic_scheduler.step()
                generator_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    model_logger.on_step_end(accelerator, model, save_steps)
                    progress_bar.set_postfix(generator_updates=(max(0, model_logger.num_steps - fake_warmup_steps) + fake_updates_per_generator - 1) // fake_updates_per_generator)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch)

    model_logger.on_training_end(accelerator, model, save_steps)
