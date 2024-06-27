from ..models import ModelManager, SVDImageEncoder, SVDUNet, SVDVAEEncoder, SVDVAEDecoder
from ..schedulers import ContinuousODEScheduler
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from einops import rearrange, repeat



class SVDVideoPipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.scheduler = ContinuousODEScheduler()
        self.device = device
        self.torch_dtype = torch_dtype
        # models
        self.image_encoder: SVDImageEncoder = None
        self.unet: SVDUNet = None
        self.vae_encoder: SVDVAEEncoder = None
        self.vae_decoder: SVDVAEDecoder = None
    

    def fetch_main_models(self, model_manager: ModelManager):
        self.image_encoder = model_manager.image_encoder
        self.unet = model_manager.unet
        self.vae_encoder = model_manager.vae_encoder
        self.vae_decoder = model_manager.vae_decoder


    @staticmethod
    def from_model_manager(model_manager: ModelManager, **kwargs):
        pipe = SVDVideoPipeline(device=model_manager.device, torch_dtype=model_manager.torch_dtype)
        pipe.fetch_main_models(model_manager)
        return pipe
    

    def preprocess_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
        return image
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        image = image.cpu().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image
    

    def encode_image_with_clip(self, image):
        image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)
        image = SVDCLIPImageProcessor().resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1).to(device=self.device, dtype=self.torch_dtype)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1).to(device=self.device, dtype=self.torch_dtype)
        image = (image - mean) / std
        image_emb = self.image_encoder(image)
        return image_emb
    

    def encode_image_with_vae(self, image, noise_aug_strength):
        image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)
        noise = torch.randn(image.shape, device="cpu", dtype=self.torch_dtype).to(self.device)
        image = image + noise_aug_strength * noise
        image_emb = self.vae_encoder(image) / self.vae_encoder.scaling_factor
        return image_emb
    

    def encode_video_with_vae(self, video):
        video = torch.concat([self.preprocess_image(frame) for frame in video], dim=0)
        video = rearrange(video, "T C H W -> 1 C T H W")
        video = video.to(device=self.device, dtype=self.torch_dtype)
        latents = self.vae_encoder.encode_video(video)
        latents = rearrange(latents[0], "C T H W -> T C H W")
        return latents
    

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    

    def calculate_noise_pred(
        self,
        latents,
        timestep,
        add_time_id,
        cfg_scales,
        image_emb_vae_posi, image_emb_clip_posi,
        image_emb_vae_nega, image_emb_clip_nega
    ):
        # Positive side
        noise_pred_posi = self.unet(
            torch.cat([latents, image_emb_vae_posi], dim=1),
            timestep, image_emb_clip_posi, add_time_id
        )
        # Negative side
        noise_pred_nega = self.unet(
            torch.cat([latents, image_emb_vae_nega], dim=1),
            timestep, image_emb_clip_nega, add_time_id
        )

        # Classifier-free guidance
        noise_pred = noise_pred_nega + cfg_scales * (noise_pred_posi - noise_pred_nega)

        return noise_pred
    

    def post_process_latents(self, latents, post_normalize=True, contrast_enhance_scale=1.0):
        if post_normalize:
            mean, std = latents.mean(), latents.std()
            latents = (latents - latents.mean(dim=[1, 2, 3], keepdim=True)) / latents.std(dim=[1, 2, 3], keepdim=True) * std + mean
        latents = latents * contrast_enhance_scale
        return latents


    @torch.no_grad()
    def __call__(
        self,
        input_image=None,
        input_video=None,
        mask_frames=[],
        mask_frame_ids=[],
        min_cfg_scale=1.0,
        max_cfg_scale=3.0,
        denoising_strength=1.0,
        num_frames=25,
        height=576,
        width=1024,
        fps=7,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        num_inference_steps=20,
        post_normalize=True,
        contrast_enhance_scale=1.2,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength)

        # Prepare latent tensors
        noise = torch.randn((num_frames, 4, height//8, width//8), device="cpu", dtype=self.torch_dtype).to(self.device)
        if denoising_strength == 1.0:
            latents = noise.clone()
        else:
            latents = self.encode_video_with_vae(input_video)
            latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])

        # Prepare mask frames
        if len(mask_frames) > 0:
            mask_latents = self.encode_video_with_vae(mask_frames)

        # Encode image
        image_emb_clip_posi = self.encode_image_with_clip(input_image)
        image_emb_clip_nega = torch.zeros_like(image_emb_clip_posi)
        image_emb_vae_posi = repeat(self.encode_image_with_vae(input_image, noise_aug_strength), "B C H W -> (B T) C H W", T=num_frames)
        image_emb_vae_nega = torch.zeros_like(image_emb_vae_posi)

        # Prepare classifier-free guidance
        cfg_scales = torch.linspace(min_cfg_scale, max_cfg_scale, num_frames)
        cfg_scales = cfg_scales.reshape(num_frames, 1, 1, 1).to(device=self.device, dtype=self.torch_dtype)
        
        # Prepare positional id
        add_time_id = torch.tensor([[fps-1, motion_bucket_id, noise_aug_strength]], device=self.device)

        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):

            # Mask frames
            for frame_id, mask_frame_id in enumerate(mask_frame_ids):
                latents[mask_frame_id] = self.scheduler.add_noise(mask_latents[frame_id], noise[mask_frame_id], timestep)

            # Fetch model output
            noise_pred = self.calculate_noise_pred(
                latents, timestep, add_time_id, cfg_scales,
                image_emb_vae_posi, image_emb_clip_posi, image_emb_vae_nega, image_emb_clip_nega
            )

            # Forward Euler
            latents = self.scheduler.step(noise_pred, timestep, latents)
            
            # Update progress bar
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))

        # Decode image
        latents = self.post_process_latents(latents, post_normalize=post_normalize, contrast_enhance_scale=contrast_enhance_scale)
        video = self.vae_decoder.decode_video(latents, progress_bar=progress_bar_cmd)
        video = self.tensor2video(video)

        return video



class SVDCLIPImageProcessor:
    def __init__(self):
        pass

    def resize_with_antialiasing(self, input, size, interpolation="bicubic", align_corners=True):
        h, w = input.shape[-2:]
        factors = (h / size[0], w / size[1])

        # First, we have to determine sigma
        # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
        sigmas = (
            max((factors[0] - 1.0) / 2.0, 0.001),
            max((factors[1] - 1.0) / 2.0, 0.001),
        )

        # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
        # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
        # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
        ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

        # Make sure it is odd
        if (ks[0] % 2) == 0:
            ks = ks[0] + 1, ks[1]

        if (ks[1] % 2) == 0:
            ks = ks[0], ks[1] + 1

        input = self._gaussian_blur2d(input, ks, sigmas)

        output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
        return output


    def _compute_padding(self, kernel_size):
        """Compute padding tuple."""
        # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
        if len(kernel_size) < 2:
            raise AssertionError(kernel_size)
        computed = [k - 1 for k in kernel_size]

        # for even kernels we need to do asymmetric padding :(
        out_padding = 2 * len(kernel_size) * [0]

        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]

            pad_front = computed_tmp // 2
            pad_rear = computed_tmp - pad_front

            out_padding[2 * i + 0] = pad_front
            out_padding[2 * i + 1] = pad_rear

        return out_padding


    def _filter2d(self, input, kernel):
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

        tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

        height, width = tmp_kernel.shape[-2:]

        padding_shape: list[int] = self._compute_padding([height, width])
        input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

        # kernel and input tensor reshape to align element-wise or batch-wise params
        tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
        input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

        # convolve the tensor with the kernel.
        output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

        out = output.view(b, c, h, w)
        return out


    def _gaussian(self, window_size: int, sigma):
        if isinstance(sigma, float):
            sigma = torch.tensor([[sigma]])

        batch_size = sigma.shape[0]

        x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

        if window_size % 2 == 0:
            x = x + 0.5

        gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

        return gauss / gauss.sum(-1, keepdim=True)


    def _gaussian_blur2d(self, input, kernel_size, sigma):
        if isinstance(sigma, tuple):
            sigma = torch.tensor([sigma], dtype=input.dtype)
        else:
            sigma = sigma.to(dtype=input.dtype)

        ky, kx = int(kernel_size[0]), int(kernel_size[1])
        bs = sigma.shape[0]
        kernel_x = self._gaussian(kx, sigma[:, 1].view(bs, 1))
        kernel_y = self._gaussian(ky, sigma[:, 0].view(bs, 1))
        out_x = self._filter2d(input, kernel_x[..., None, :])
        out = self._filter2d(out_x, kernel_y[..., None])

        return out
