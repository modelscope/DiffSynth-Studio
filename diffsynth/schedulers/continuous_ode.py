import torch


class ContinuousODEScheduler():

    def __init__(self, num_inference_steps=100, sigma_max=700.0, sigma_min=0.002, rho=7.0):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, **kwargs):
        ramp = torch.linspace(1-denoising_strength, 1, num_inference_steps)
        min_inv_rho = torch.pow(torch.tensor((self.sigma_min,)), (1 / self.rho))
        max_inv_rho = torch.pow(torch.tensor((self.sigma_max,)), (1 / self.rho))
        self.sigmas = torch.pow(max_inv_rho + ramp * (min_inv_rho - max_inv_rho), self.rho)
        self.timesteps = torch.log(self.sigmas) * 0.25


    def step(self, model_output, timestep, sample, to_final=False):
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample *= (sigma*sigma + 1).sqrt()
        estimated_sample = -sigma / (sigma*sigma + 1).sqrt() * model_output + 1 / (sigma*sigma + 1) * sample
        if to_final or timestep_id + 1 >= len(self.timesteps):
            prev_sample = estimated_sample
        else:
            sigma_ = self.sigmas[timestep_id + 1]
            derivative = 1 / sigma * (sample - estimated_sample)
            prev_sample = sample + derivative * (sigma_ - sigma)
            prev_sample /= (sigma_*sigma_ + 1).sqrt()
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        # This scheduler doesn't support this function.
        pass
    
    
    def add_noise(self, original_samples, noise, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (original_samples + noise * sigma) / (sigma*sigma + 1).sqrt()
        return sample
    

    def training_target(self, sample, noise, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        target = (-(sigma*sigma + 1).sqrt() / sigma + 1 / (sigma*sigma + 1).sqrt() / sigma) * sample + 1 / (sigma*sigma + 1).sqrt() * noise
        return target
    

    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        weight = (1 + sigma*sigma).sqrt() / sigma
        return weight
