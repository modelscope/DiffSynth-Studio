import torch
import numpy as np
from PIL import Image
from torchvision.transforms import GaussianBlur



class BasePipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16, height_division_factor=64, width_division_factor=64):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.cpu_offload = False
        self.model_names = []


    def check_resize_height_width(self, height, width):
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"The height cannot be evenly divided by {self.height_division_factor}. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"The width cannot be evenly divided by {self.width_division_factor}. We round it up to {width}.")
        return height, width


    def preprocess_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
        return image
    

    def preprocess_images(self, images):
        return [self.preprocess_image(image) for image in images]
    

    def vae_output_to_image(self, vae_output):
        image = vae_output[0].cpu().float().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image
    

    def vae_output_to_video(self, vae_output):
        video = vae_output.cpu().permute(1, 2, 0).numpy()
        video = [Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8")) for image in video]
        return video

    
    def merge_latents(self, value, latents, masks, scales, blur_kernel_size=33, blur_sigma=10.0):
        if len(latents) > 0:
            blur = GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
            height, width = value.shape[-2:]
            weight = torch.ones_like(value)
            for latent, mask, scale in zip(latents, masks, scales):
                mask = self.preprocess_image(mask.resize((width, height))).mean(dim=1, keepdim=True) > 0
                mask = mask.repeat(1, latent.shape[1], 1, 1).to(dtype=latent.dtype, device=latent.device)
                mask = blur(mask)
                value += latent * mask * scale
                weight += mask * scale
            value /= weight
        return value


    def control_noise_via_local_prompts(self, prompt_emb_global, prompt_emb_locals, masks, mask_scales, inference_callback, special_kwargs=None, special_local_kwargs_list=None):
        if special_kwargs is None:
            noise_pred_global = inference_callback(prompt_emb_global)
        else:
            noise_pred_global = inference_callback(prompt_emb_global, special_kwargs)
        if special_local_kwargs_list is None:
            noise_pred_locals = [inference_callback(prompt_emb_local) for prompt_emb_local in prompt_emb_locals]
        else:
            noise_pred_locals = [inference_callback(prompt_emb_local, special_kwargs) for prompt_emb_local, special_kwargs in zip(prompt_emb_locals, special_local_kwargs_list)]
        noise_pred = self.merge_latents(noise_pred_global, noise_pred_locals, masks, mask_scales)
        return noise_pred
    

    def extend_prompt(self, prompt, local_prompts, masks, mask_scales):
        local_prompts = local_prompts or []
        masks = masks or []
        mask_scales = mask_scales or []
        extended_prompt_dict = self.prompter.extend_prompt(prompt)
        prompt = extended_prompt_dict.get("prompt", prompt)
        local_prompts += extended_prompt_dict.get("prompts", [])
        masks += extended_prompt_dict.get("masks", [])
        mask_scales += [100.0] * len(extended_prompt_dict.get("masks", []))
        return prompt, local_prompts, masks, mask_scales
    
    
    def enable_cpu_offload(self):
        self.cpu_offload = True


    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    
    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise
