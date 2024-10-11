import torch, math


class EnhancedDDIMScheduler():

    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", prediction_type="epsilon", rescale_zero_terminal_snr=False):
        self.num_train_timesteps = num_train_timesteps
        if beta_schedule == "scaled_linear":
            betas = torch.square(torch.linspace(math.sqrt(beta_start), math.sqrt(beta_end), num_train_timesteps, dtype=torch.float32))
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented")
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        if rescale_zero_terminal_snr:
            self.alphas_cumprod = self.rescale_zero_terminal_snr(self.alphas_cumprod)
        self.alphas_cumprod = self.alphas_cumprod.tolist()
        self.set_timesteps(10)
        self.prediction_type = prediction_type


    def rescale_zero_terminal_snr(self, alphas_cumprod):
        alphas_bar_sqrt = alphas_cumprod.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt.square()  # Revert sqrt

        return alphas_bar


    def set_timesteps(self, num_inference_steps, denoising_strength=1.0, **kwargs):
        # The timesteps are aligned to 999...0, which is different from other implementations,
        # but I think this implementation is more reasonable in theory.
        max_timestep = max(round(self.num_train_timesteps * denoising_strength) - 1, 0)
        num_inference_steps = min(num_inference_steps, max_timestep + 1)
        if num_inference_steps == 1:
            self.timesteps = torch.Tensor([max_timestep])
        else:
            step_length = max_timestep / (num_inference_steps - 1)
            self.timesteps = torch.Tensor([round(max_timestep - i*step_length) for i in range(num_inference_steps)])


    def denoise(self, model_output, sample, alpha_prod_t, alpha_prod_t_prev):
        if self.prediction_type == "epsilon":
            weight_e = math.sqrt(1 - alpha_prod_t_prev) - math.sqrt(alpha_prod_t_prev * (1 - alpha_prod_t) / alpha_prod_t)
            weight_x = math.sqrt(alpha_prod_t_prev / alpha_prod_t)
            prev_sample = sample * weight_x + model_output * weight_e
        elif self.prediction_type == "v_prediction":
            weight_e = -math.sqrt(alpha_prod_t_prev * (1 - alpha_prod_t)) + math.sqrt(alpha_prod_t * (1 - alpha_prod_t_prev))
            weight_x = math.sqrt(alpha_prod_t * alpha_prod_t_prev) + math.sqrt((1 - alpha_prod_t) * (1 - alpha_prod_t_prev))
            prev_sample = sample * weight_x + model_output * weight_e
        else:
            raise NotImplementedError(f"{self.prediction_type} is not implemented")
        return prev_sample


    def step(self, model_output, timestep, sample, to_final=False):
        alpha_prod_t = self.alphas_cumprod[int(timestep.flatten().tolist()[0])]
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        if to_final or timestep_id + 1 >= len(self.timesteps):
            alpha_prod_t_prev = 1.0
        else:
            timestep_prev = int(self.timesteps[timestep_id + 1])
            alpha_prod_t_prev = self.alphas_cumprod[timestep_prev]

        return self.denoise(model_output, sample, alpha_prod_t, alpha_prod_t_prev)


    def return_to_timestep(self, timestep, sample, sample_stablized):
        alpha_prod_t = self.alphas_cumprod[int(timestep.flatten().tolist()[0])]
        noise_pred = (sample - math.sqrt(alpha_prod_t) * sample_stablized) / math.sqrt(1 - alpha_prod_t)
        return noise_pred
    
    
    def add_noise(self, original_samples, noise, timestep):
        sqrt_alpha_prod = math.sqrt(self.alphas_cumprod[int(timestep.flatten().tolist()[0])])
        sqrt_one_minus_alpha_prod = math.sqrt(1 - self.alphas_cumprod[int(timestep.flatten().tolist()[0])])
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    

    def training_target(self, sample, noise, timestep):
        if self.prediction_type == "epsilon":
            return noise
        else:
            sqrt_alpha_prod = math.sqrt(self.alphas_cumprod[int(timestep.flatten().tolist()[0])])
            sqrt_one_minus_alpha_prod = math.sqrt(1 - self.alphas_cumprod[int(timestep.flatten().tolist()[0])])
            target = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
            return target
        
    
    def training_weight(self, timestep):
        return 1.0
