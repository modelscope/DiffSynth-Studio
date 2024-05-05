import torch, math


class ContinuousODEScheduler():

    def __init__(self, num_inference_steps=100, sigma_max=700.0, sigma_min=0.002, rho=7.0):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.init_noise_sigma = math.sqrt(sigma_max*sigma_max + 1)
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0):
        ramp = torch.linspace(0, denoising_strength, num_inference_steps)
        min_inv_rho = torch.pow(torch.tensor((self.sigma_min,)), (1 / self.rho))
        max_inv_rho = torch.pow(torch.tensor((self.sigma_max,)), (1 / self.rho))
        self.sigmas = torch.pow(max_inv_rho + ramp * (min_inv_rho - max_inv_rho), self.rho)
        self.timesteps = torch.log(self.sigmas) * 0.25


    def step(self, model_output, timestep, sample, to_final=False):
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        estimated_sample = -sigma / (sigma*sigma + 1).sqrt() * model_output + 1 / (sigma*sigma + 1) * sample
        if to_final or timestep_id + 1 >= len(self.timesteps):
            prev_sample = estimated_sample
        else:
            dt = self.sigmas[timestep_id + 1] - sigma
            derivative = 1 / sigma * (sample - estimated_sample)
            prev_sample = sample + derivative * dt
        return prev_sample
    
    
    def scale_model_input(self, sample, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = sample / (sigma*sigma + 1).sqrt()
        return sample


    def return_to_timestep(self, timestep, sample, sample_stablized):
        # This scheduler doesn't support this function.
        pass
    
    
    def add_noise(self, original_samples, noise, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = original_samples + noise * sigma
        return sample

