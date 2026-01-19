import torch, math
from typing_extensions import Literal, Optional, List


class FlowMatchScheduler():

    def __init__(self, template: Literal["FLUX.1", "Wan", "Qwen-Image", "FLUX.2", "Z-Image"] = "FLUX.1"):
        self.set_timesteps_fn = {
            "FLUX.1": FlowMatchScheduler.set_timesteps_flux,
            "Wan": FlowMatchScheduler.set_timesteps_wan,
            "Qwen-Image": FlowMatchScheduler.set_timesteps_qwen_image,
            "FLUX.2": FlowMatchScheduler.set_timesteps_flux2,
            "Z-Image": FlowMatchScheduler.set_timesteps_z_image,
        }.get(template, FlowMatchScheduler.set_timesteps_flux)
        self.num_train_timesteps = 1000

    @staticmethod
    def set_timesteps_flux(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.003/1.002
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def _calculate_shift_qwen_image(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
    
    @staticmethod
    def set_timesteps_qwen_image(num_inference_steps=100, denoising_strength=1.0, exponential_shift_mu=None, dynamic_shift_len=None):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        shift_terminal = 0.02
        # Sigmas
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        # Mu
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = FlowMatchScheduler._calculate_shift_qwen_image(dynamic_shift_len)
        else:
            mu = 0.8
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        # Shift terminal
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - shift_terminal)
        sigmas = 1 - (one_minus_z / scale_factor)
        # Timesteps
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def compute_empirical_mu(image_seq_len, num_steps):
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)
    
    @staticmethod
    def set_timesteps_flux2(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=1024//16*1024//16):
        sigma_min = 1 / num_inference_steps
        sigma_max = 1.0
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        mu = FlowMatchScheduler.compute_empirical_mu(dynamic_shift_len, num_inference_steps)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_z_image(num_inference_steps=100, denoising_strength=1.0, shift=None, target_timesteps=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        if target_timesteps is not None:
            target_timesteps = target_timesteps.to(dtype=timesteps.dtype, device=timesteps.device)
            for timestep in target_timesteps:
                timestep_id = torch.argmin((timesteps - timestep).abs())
                timesteps[timestep_id] = timestep
        return sigmas, timesteps
    
    def set_training_weight(self):
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing
        
    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    
    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    
    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    
    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights


class FlowMatchSDEScheduler(FlowMatchScheduler):

    def __init__(self,
        template : Literal["FLUX.1", "Wan", "Qwen-Image", "FLUX.2", "Z-Image"] = "FLUX.1",
        noise_level : float = 0.1,
        noise_window : Optional[List[int]] = None,
        sde_step_num : Optional[int] = None,
        sde_type : Literal['Flow-SDE', 'Dance-SDE', 'CPS'] = 'Flow-SDE',
        seed: Optional[int] = None,
        **kwargs,
        ):
        super().__init__(template=template, **kwargs)
        if noise_window is None:
            self.noise_window = list(range(self.num_train_timesteps))
        else:
            self.noise_window = list(noise_window)

        self.noise_level = noise_level
        self.sde_step_num = sde_step_num or len(self.noise_window)
        self.sde_type = sde_type
        self.seed = seed or 42

    def set_seed(self, seed: int) -> torch.Tensor:
        self.seed = seed

    @property
    def current_noise_steps(self) -> torch.Tensor:
        if self.sde_step_num >= len(self.noise_window):
            return self.noise_window
        generator = torch.Generator().manual_seed(self.seed)
        selected_indices = torch.randperm(len(self.noise_window), generator=generator)[:self.sde_step_num]
        return self.noise_window[selected_indices]

    def get_current_noise_level(self, timestep) -> float:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        if timestep_id in self.current_noise_steps:
            return self.noise_level
        else:
            return 0.0

    def step(self,
             model_output,
             timestep,
             sample,
             to_final=False,
             prev_sample=None,
             generator=None,
             return_log_prob=False,
             return_dict=False,
             **kwargs
        ):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]

        # Convert to float32 for numerical stability
        model_output = model_output.float()
        sample = sample.float()

        current_noise_level = self.get_current_noise_level(timestep)

        dt = sigma_ - sigma
        sigma_max = self.sigmas[1] # Use the max sigma < 1
        if self.sde_type == 'Flow-SDE':
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * current_noise_level
            prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            if prev_sample is None:
                variance_noise = torch.randn_like(
                    model_output.shape,
                    generator=generator,
                ).to(device=model_output.device, dtype=model_output.dtype)
                prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

            if return_log_prob:
                log_prob = (
                    -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
                    - torch.log(std_dev_t * torch.sqrt(-1 * dt))
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                )
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        elif self.sde_type == 'Dance-SDE':
            pred_original_sample = sample - sigma * model_output
            std_dev_t = current_noise_level * torch.sqrt(-1 * dt)
            log_term = 0.5 * current_noise_level**2 * (sample - pred_original_sample * (1 - sigma)) / sigma**2
            prev_sample_mean = sample + (model_output + log_term) * dt
            if prev_sample is None:
                variance_noise = torch.randn(
                    model_output.shape,
                    generator=generator,
                ).to(device=model_output.device, dtype=model_output.dtype)
                prev_sample = prev_sample_mean + std_dev_t * variance_noise
            
            if return_log_prob:
                log_prob = (
                    (-((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2)))
                    - math.log(std_dev_t)
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                )

                # mean along all but batch dimension
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        elif self.sde_type == 'CPS':
            # Coefficient Preserving Sampling
            std_dev_t = sigma_ * torch.sin(current_noise_level * torch.pi / 2)
            pred_original_sample = sample - sigma * model_output
            noise_estimate = sample + model_output * (1 - sigma)
            prev_sample_mean = pred_original_sample * (1 - sigma_) + noise_estimate * torch.sqrt(sigma_**2 - std_dev_t**2)
            
            if prev_sample is None:
                variance_noise = torch.randn(
                    model_output.shape,
                    generator=generator,
                ).to(device=model_output.device, dtype=model_output.dtype)
                prev_sample = prev_sample_mean + std_dev_t * variance_noise

            if return_log_prob:
                log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        if not return_log_prob:
            log_prob = torch.zeros(sample.shape[0], device=sample.device)
        
        if return_dict:
            return {
                "prev_sample": prev_sample,
                "log_prob": log_prob,
                "prev_sample_mean": prev_sample_mean,
                "std_dev_t": std_dev_t,
            }
        
        return prev_sample, log_prob, prev_sample_mean, std_dev_t