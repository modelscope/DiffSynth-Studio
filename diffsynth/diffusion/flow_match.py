import torch, math
import numpy as np
from typing_extensions import Literal


class FlowMatchScheduler():

    def __init__(self, template: Literal["FLUX.1", "Wan", "Qwen-Image", "FLUX.2", "Z-Image", "LTX-2", "Qwen-Image-Lightning", "ERNIE-Image", "ACE-Step", "Ideogram4", "Krea-2", "Boogu"] = "FLUX.1"):
        self.set_timesteps_fn = {
            "FLUX.1": FlowMatchScheduler.set_timesteps_flux,
            "Wan": FlowMatchScheduler.set_timesteps_wan,
            "Qwen-Image": FlowMatchScheduler.set_timesteps_qwen_image,
            "FLUX.2": FlowMatchScheduler.set_timesteps_flux2,
            "Z-Image": FlowMatchScheduler.set_timesteps_z_image,
            "LTX-2": FlowMatchScheduler.set_timesteps_ltx2,
            "Qwen-Image-Lightning": FlowMatchScheduler.set_timesteps_qwen_image_lightning,
            "ERNIE-Image": FlowMatchScheduler.set_timesteps_ernie_image,
            "ACE-Step": FlowMatchScheduler.set_timesteps_ace_step,
            "HiDream-O1-Image": FlowMatchScheduler.set_timesteps_hidream_o1_image,
            "Ideogram4": FlowMatchScheduler.set_timesteps_ideogram4,
            "Krea-2": FlowMatchScheduler.set_timesteps_krea2,
            "Boogu": FlowMatchScheduler.set_timesteps_boogu,
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
    def set_timesteps_qwen_image_lightning(num_inference_steps=100, denoising_strength=1.0, exponential_shift_mu=None, dynamic_shift_len=None):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        base_shift = math.log(3)
        max_shift = math.log(3)
        # Sigmas
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        # Mu
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = FlowMatchScheduler._calculate_shift_qwen_image(dynamic_shift_len, base_shift=base_shift, max_shift=max_shift)
        else:
            mu = 0.8
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
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
    def set_timesteps_flux2(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=None):
        sigma_min = 1 / num_inference_steps
        sigma_max = 1.0
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        if dynamic_shift_len is None:
            # If you ask me why I set mu=0.8,
            # I can only say that it yields better training results.
            mu = 0.8
        else:
            mu = FlowMatchScheduler.compute_empirical_mu(dynamic_shift_len, num_inference_steps)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_ernie_image(num_inference_steps=50, denoising_strength=1.0, shift=3.0):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        if shift is not None and shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_ace_step(num_inference_steps=8, denoising_strength=1.0, shift=3.0):
        num_train_timesteps = 1000
        sigma_start = denoising_strength
        sigmas = torch.linspace(sigma_start, 0.0, num_inference_steps + 1)[:-1]
        if shift is not None and shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
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

    @staticmethod
    def set_timesteps_joyai_image(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 4.0 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_hidream_o1_image(num_inference_steps=28, denoising_strength=1.0, shift=None, special_case=None, **kwargs):
        num_train_timesteps = 1000
        shift = 3.0 if shift is None else shift
        if special_case == "dev":
            timesteps_list = [
                999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
                707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
            ]
            sigmas = torch.tensor([t / 1000.0 for t in timesteps_list], dtype=torch.float32)
            timesteps = torch.tensor(timesteps_list, dtype=torch.float32)
            return sigmas, timesteps
        else:
            sigma_min = 0.0
            sigma_max = 1.0
            sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
            sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            timesteps = sigmas * num_train_timesteps
            return sigmas, timesteps

    @staticmethod
    def set_timesteps_ideogram4(num_inference_steps=50, denoising_strength=1.0, image_resolution=(1024, 1024), mu=0.0, std=1.5):
        num_pixels = image_resolution[0] * image_resolution[1]
        known_pixels = 512 * 512
        mean = mu + 0.5 * math.log(num_pixels / known_pixels)
        logsnr_min = -15.0
        logsnr_max = 18.0
        t_min = 1.0 / (1 + math.exp(0.5 * logsnr_max))
        t_max = 1.0 / (1 + math.exp(0.5 * logsnr_min))
        step_intervals = torch.linspace(0.0, denoising_strength, num_inference_steps + 1, dtype=torch.float64)
        sigmas = []
        for i in range(num_inference_steps + 1):
            z = torch.special.ndtri(step_intervals[i])
            y = mean + std * z
            t_ = torch.special.expit(y)
            t_ = 1 - t_
            t_ = t_.clamp(t_min, t_max)
            sigmas.append(float(t_.to(torch.float32)))
        sigmas = torch.tensor(sigmas, dtype=torch.float32)
        one_minus_t = (1 - sigmas)[:-1].flip(0)
        sigma_start = one_minus_t[0] * denoising_strength
        if one_minus_t[0] > 0:
            one_minus_t = one_minus_t * (sigma_start / one_minus_t[0])
        sigmas = sigmas.flip(dims=(0,))
        timesteps = sigmas[:-1]
        sigmas = (1 - sigmas)[:-1]
        return sigmas, timesteps
    
    def set_timesteps_boogu(num_inference_steps=50, denoising_strength=1.0, sigmas=None):
        if sigmas is not None:
            sigmas = torch.tensor(sigmas, dtype=torch.float32)
            timesteps = 1 - sigmas
            return sigmas, timesteps
        t_arr = np.linspace(1-denoising_strength, 1, num_inference_steps + 1, dtype=np.float32)
        mu = 1.15
        sigma = 1
        eps = 1e-8
        t1 = 1.0 - t_arr
        t1 = np.clip(t1, eps, 1.0 - eps)
        num = math.exp(mu)
        denom = num + np.power(1.0 / t1 - 1.0, sigma)
        y = num / denom
        t_arr = 1.0 - y
        timesteps = torch.from_numpy(t_arr).float()[:-1]
        sigmas = 1 - timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_krea2(num_inference_steps=28, denoising_strength=1.0, dynamic_shift_len=None, y1=0.5, y2=1.15, mu=None):
        x1 = 256
        x2 = 6400
        sigma = 1
        ts = torch.linspace(denoising_strength, 0, num_inference_steps + 1)
        if mu is None and dynamic_shift_len is None:
            # Training
            mu = 0.8
        elif mu is None:
            # Raw
            slope = (y2 - y1) / (x2 - x1)
            mu = slope * dynamic_shift_len + (y1 - slope * x1)
        ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
        sigmas, timesteps = ts[:-1], ts[:-1]
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_ltx2(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=None, terminal=0.1, special_case=None):
        num_train_timesteps = 1000
        if special_case == "stage2":
            sigmas = torch.Tensor([0.909375, 0.725, 0.421875])
        elif special_case == "ditilled_stage1":
            sigmas = torch.Tensor([1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875])
        else:
            dynamic_shift_len = dynamic_shift_len or 4096
            sigma_shift = FlowMatchScheduler._calculate_shift_qwen_image(
                image_seq_len=dynamic_shift_len,
                base_seq_len=1024,
                max_seq_len=4096,
                base_shift=0.95,
                max_shift=2.05,
            )
            sigma_min = 0.0
            sigma_max = 1.0
            sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
            sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
            sigmas = math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1))
            # Shift terminal
            one_minus_z = 1.0 - sigmas
            scale_factor = one_minus_z[-1] / (1 - terminal)
            sigmas = 1.0 - (one_minus_z / scale_factor)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_training_weight(self):
        steps = 1000
        x = self.sigmas * self.num_train_timesteps
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


class HiDreamO1FlashScheduler(FlowMatchScheduler):
    
    def __init__(self, noise_scale_start=7.5, noise_scale_end=7.5, noise_clip_std=2.5):
        self.set_timesteps_fn = HiDreamO1FlashScheduler.set_timesteps_hidream_o1_image_dev
        self.num_train_timesteps = 1000
        self.noise_clip_std = noise_clip_std
        num_steps = 28
        self.noise_scale_schedule = [
            noise_scale_start + (noise_scale_end - noise_scale_start) * i / (num_steps - 1)
            for i in range(num_steps)
        ]

    @staticmethod
    def set_timesteps_hidream_o1_image_dev(**kwargs):
        timesteps_list = [
            999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
            707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
        ]
        sigmas = torch.tensor([t / 1000.0 for t in timesteps_list], dtype=torch.float32)
        timesteps = torch.tensor(timesteps_list, dtype=torch.float32)
        return sigmas, timesteps

    def clip_noise(self, noise):
        if self.noise_clip_std > 0:
            noise_std = noise.std().item()
            clip_val = self.noise_clip_std * noise_std
            noise = noise.clamp(min=-clip_val, max=clip_val)
        return noise
    
    def step(self, model_output, timestep, sample):
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sigma_ = self.sigmas[timestep_id + 1] if timestep_id + 1 < len(self.sigmas) else 0
        denoised = sample - model_output * sigma

        noise = self.clip_noise(torch.randn(denoised.shape, device=denoised.device, dtype=denoised.dtype))
        sample = sigma_ * noise * self.noise_scale_schedule[timestep_id] + (1.0 - sigma_) * denoised
        return sample


class LingBotVideoUniPCScheduler(FlowMatchScheduler):
    """
    UniPC multistep predictor-corrector scheduler for flow-matching, ported from
    LingBot-Video's ``FlowUniPCMultistepScheduler`` (a vendored diffusers UniPC
    variant, ``prediction_type="flow_prediction"``).

    This scheduler is used for **inference** sampling only. Its ``step`` implements
    the stateful multistep UniP predictor + UniC corrector. Training reuses the
    flow-matching interpolation (``add_noise``) and velocity target
    (``training_target`` / ``training_weight``) inherited from ``FlowMatchScheduler``
    — UniPC is a sampler, not a training objective.

    Base-sigma grid matches the original exactly: ``sigma = 1 - linspace(1, 1/N, N)[::-1]``
    (i.e. ``sigma in [0, 1-1/N]``), which is offset by one position from the native
    diffusers UniPC grid and is required for numerical parity with LingBot-Video.
    """
    order = 1

    def __init__(
        self,
        shift=1.0,
        num_train_timesteps=1000,
        solver_order=2,
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        disable_corrector=(),
        final_sigmas_type="zero",
    ):
        if solver_type not in ("bh1", "bh2"):
            raise NotImplementedError(f"solver_type={solver_type} is not implemented")
        if final_sigmas_type not in ("zero", "sigma_min"):
            raise ValueError("final_sigmas_type must be one of 'zero' or 'sigma_min'")
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.solver_order = solver_order
        self.predict_x0 = predict_x0
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.disable_corrector = list(disable_corrector)
        self.final_sigmas_type = final_sigmas_type
        self.prediction_type = "flow_prediction"
        self.num_inference_steps = None
        self.training = False
        # base sigma grid (drives sigma_min / sigma_max and the training schedule)
        self.sigmas, self.timesteps = self._flow_sigmas(num_train_timesteps, shift)
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self._reset_multistep_state()

    def _flow_sigmas(self, n, shift):
        # sigma = 1 - alpha, alpha = linspace(1, 1/n, n)[::-1]  ->  sigma in [0, 1-1/n]
        alphas = np.linspace(1, 1 / n, n)[::-1].copy()
        sigmas = torch.from_numpy(1.0 - alphas).to(dtype=torch.float32)
        if shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * n
        return sigmas, timesteps

    def _reset_multistep_state(self):
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = None
        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index=0):
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0, shift=None, training=False, **kwargs):
        if shift is None:
            shift = self.shift
        if training:
            # Flow-matching training schedule (UniPC is inference-only): full-resolution
            # sigma grid + per-timestep loss weights from the base class.
            self.sigmas, self.timesteps = self._flow_sigmas(self.num_train_timesteps, shift)
            self.set_training_weight()
            self.training = True
            return
        self.training = False
        # Inference grid. denoising_strength<1 starts partway down the chain
        # (DiffSynth convention), matching the original at strength=1.0.
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        sigmas = np.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1].copy()
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        sigma_last = 0.0 if self.final_sigmas_type == "zero" else self.sigma_min
        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)
        # int64 timesteps match the values the original model was conditioned on
        self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.int64)
        self.num_inference_steps = len(timesteps)
        self._reset_multistep_state()

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def convert_model_output(self, model_output, sample):
        # prediction_type == "flow_prediction"
        sigma_t = self.sigmas[self.step_index]
        if self.predict_x0:
            return sample - sigma_t * model_output
        return sample - (1 - sigma_t) * model_output

    def multistep_uni_p_bh_update(self, model_output, sample, order):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = sample

        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = sample.device

        rks, D1s = [], []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R, b = [], []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = hh if self.solver_type == "bh1" else torch.expm1(hh)
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s) if D1s is not None else 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s) if D1s is not None else 0
            x_t = x_t_ - sigma_t * B_h * pred_res
        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(self, this_model_output, last_sample, this_sample, order):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = this_sample.device

        rks, D1s = [], []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R, b = [], []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = hh if self.solver_type == "bh1" else torch.expm1(hh)
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        D1s = torch.stack(D1s, dim=1) if len(D1s) > 0 else None
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s) if D1s is not None else 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s) if D1s is not None else 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_t.to(x.dtype)

    def step(self, model_output, timestep, sample, **kwargs):
        if self.num_inference_steps is None:
            raise ValueError("Run set_timesteps() before step().")
        if self.step_index is None:
            self._init_step_index(timestep)

        use_corrector = (
            self.step_index > 0
            and self.step_index - 1 not in self.disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output_convert, sample=sample, order=self.this_order,
        )
        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1
        return prev_sample
