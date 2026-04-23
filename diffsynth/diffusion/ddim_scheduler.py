import torch, math
from typing import Literal


class DDIMScheduler:

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2"] = "scaled_linear",
        clip_sample: bool = False,
        set_alpha_to_one: bool = False,
        steps_offset: int = 1,
        prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon",
        timestep_spacing: Literal["leading", "trailing", "linspace"] = "leading",
        rescale_betas_zero_snr: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.set_alpha_to_one = set_alpha_to_one
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing

        # Compute betas
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # SD 1.5 specific: sqrt-linear interpolation
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = self._betas_for_alpha_bar(num_train_timesteps)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas = self._rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # For the final step, there is no previous alphas_cumprod
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # Setable values (will be populated by set_timesteps)
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(self._default_timesteps().astype("int64"))
        self.training = False

    @staticmethod
    def _betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
        """Create beta schedule via cosine alpha_bar function."""
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    @staticmethod
    def _rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
        """Rescale betas to have zero terminal SNR."""
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = alphas_cumprod.sqrt()
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
        alphas_bar = alphas_bar_sqrt ** 2
        alphas = torch.cat([alphas_bar[1:], alphas_bar[:1]])
        return 1 - alphas

    def _default_timesteps(self):
        """Default timesteps before set_timesteps is called."""
        import numpy as np
        return np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)

    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        """Compute the variance for the DDIM step."""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def set_timesteps(self, num_inference_steps: int = 100, denoising_strength: float = 1.0, training: bool = False, **kwargs):
        """
        Sets the discrete timesteps used for the diffusion chain.
        Follows FlowMatchScheduler interface: (num_inference_steps, denoising_strength, training, **kwargs)
        """
        import numpy as np

        if denoising_strength != 1.0:
            # For img2img: adjust effective steps
            num_inference_steps = int(num_inference_steps * denoising_strength)

        # Compute step ratio
        if self.timestep_spacing == "leading":
            # leading: arange * step_ratio, reverse, then add offset
            step_ratio = self.num_train_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
            timesteps = timesteps + self.steps_offset
        elif self.timestep_spacing == "trailing":
            # trailing: timesteps = arange(num_steps, 0, -1) * step_ratio - 1
            step_ratio = self.num_train_timesteps / num_inference_steps
            timesteps = (np.arange(num_inference_steps, 0, -1) * step_ratio - 1).round()[::-1]
        elif self.timestep_spacing == "linspace":
            # linspace: evenly spaced from num_train_timesteps - 1 to 0
            timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps).round()[::-1]
        else:
            raise ValueError(f"Unsupported timestep_spacing: {self.timestep_spacing}")

        self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.int64)
        self.num_inference_steps = num_inference_steps

        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def set_training_weight(self):
        """Set timestep weights for training (similar to FlowMatchScheduler)."""
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing

    def step(self, model_output, timestep, sample, to_final: bool = False, eta: float = 0.0, **kwargs):
        """
        DDIM step function.
        Follows FlowMatchScheduler interface: step(model_output, timestep, sample, to_final=False)

        For SD 1.5, prediction_type="epsilon" and eta=0.0 (deterministic DDIM).
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
            if timestep.dim() == 0:
                timestep = timestep.item()
            elif timestep.dim() == 1:
                timestep = timestep[0].item()

        # Ensure timestep is int
        timestep = int(timestep)

        # Find the index of the current timestep
        timestep_id = torch.argmin((self.timesteps - timestep).abs()).item()

        if timestep_id + 1 >= len(self.timesteps):
            prev_timestep = -1
        else:
            prev_timestep = self.timesteps[timestep_id + 1].item()

        # Get alphas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        alpha_prod_t = alpha_prod_t.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t_prev = alpha_prod_t_prev.to(device=sample.device, dtype=sample.dtype)

        beta_prod_t = 1 - alpha_prod_t

        # Compute predicted original sample (x_0)
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        # Clip sample if needed
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)

        # Compute predicted noise (re-derived from x_0)
        pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()

        # DDIM formula: prev_sample = sqrt(alpha_prev) * x0 + sqrt(1 - alpha_prev) * epsilon
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + (1 - alpha_prod_t_prev).sqrt() * pred_epsilon

        # Add variance noise if eta > 0 (DDIM: eta=0, DDPM: eta=1)
        if eta > 0:
            variance = self._get_variance(timestep, prev_timestep)
            variance = variance.to(device=sample.device, dtype=sample.dtype)
            std_dev_t = eta * variance.sqrt()
            device = sample.device
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + std_dev_t * noise

        return prev_sample

    def add_noise(self, original_samples, noise, timestep):
        """Add noise to original samples (forward diffusion).
        Follows FlowMatchScheduler interface: add_noise(original_samples, noise, timestep)
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
            if timestep.dim() == 0:
                timestep = timestep.item()
            elif timestep.dim() == 1:
                timestep = timestep[0].item()

        timestep = int(timestep)
        sqrt_alpha_prod = self.alphas_cumprod[timestep].sqrt()
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timestep]).sqrt()

        sqrt_alpha_prod = sqrt_alpha_prod.to(device=original_samples.device, dtype=original_samples.dtype)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to(device=original_samples.device, dtype=original_samples.dtype)

        # Handle broadcasting for batch timesteps
        while sqrt_alpha_prod.dim() < original_samples.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        sample = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return sample

    def training_target(self, sample, noise, timestep):
        """Return the training target for the given prediction type."""
        if self.prediction_type == "epsilon":
            return noise
        elif self.prediction_type == "v_prediction":
            sqrt_alpha_prod = self.alphas_cumprod[timestep].sqrt()
            sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timestep]).sqrt()
            return sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        elif self.prediction_type == "sample":
            return sample
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

    def training_weight(self, timestep):
        """Return training weight for the given timestep."""
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        return self.linear_timesteps_weights[timestep_id]
