from ..models import ModelManager
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import WanLayerNorm, WanRMSNorm
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample



class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16
        
        # TeaCache initialization
        self.enable_teacache = False
        self.teacache_thresh = 0.05
        self.teacache_cnt = 0
        self.teacache_num_steps = 50
        self.teacache_accumulated_rel_l1_distance = 0
        self.teacache_previous_modulated_input = None
        self.teacache_previous_residual_cond = None
        self.teacache_previous_residual_uncond = None
        self.teacache_should_calc = True
        self.teacache_coefficients = None
        self.teacache_first_step = True  # Track first step explicitly


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        
        # Set TeaCache coefficients based on model size
        if self.dit is not None:
            model_params = sum(p.numel() for p in self.dit.parameters())
            if model_params > 10e9:  # Large model ~14B
                self.teacache_coefficients = [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404]
            else:  # Small model ~1.3B
                self.teacache_coefficients = [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01]


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, num_frames, height, width):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            image = self.preprocess_image(image.resize((width, height))).to(self.device)
            clip_context = self.image_encoder.encode_image([image])
            msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = self.vae.encode([torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)], device=self.device)[0]
            y = torch.concat([msk, y])
        return {"clip_fea": clip_context, "y": [y]}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {"seq_len": latents.shape[2] * latents.shape[3] * latents.shape[4] // 4}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    
    def setup_teacache(self, enable=False, thresh=0.05, num_steps=50, model_size=None):
        """
        Set up the TeaCache optimization.
        
        Args:
            enable (bool): Whether to enable TeaCache optimization
            thresh (float): Threshold for cumulative relative L1 distance to trigger recalculation
            num_steps (int): Total number of denoising steps
            model_size (str): Model size identifier for selecting appropriate coefficients
        """
        self.enable_teacache = enable
        self.teacache_thresh = thresh
        self.teacache_num_steps = num_steps
        self.teacache_cnt = 0
        self.teacache_accumulated_rel_l1_distance = 0
        self.teacache_previous_modulated_input = None
        self.teacache_previous_residual_cond = None
        self.teacache_previous_residual_uncond = None
        self.teacache_should_calc = True
        self.teacache_first_step = True
        
        # Set coefficients based on model size if provided
        if model_size is not None:
            if model_size == "14B":
                self.teacache_coefficients = [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404]
            elif model_size == "1.3B":
                self.teacache_coefficients = [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
            elif model_size == "480P":
                self.teacache_coefficients = [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01]
            elif model_size == "720P":
                self.teacache_coefficients = [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683]
        
        print(f"TeaCache {'enabled' if enable else 'disabled'} with threshold {thresh} for {num_steps} steps")


    def apply_teacache(self, latents, timestep, is_cond, **kwargs):
        """
        Apply the TeaCache optimization to the model forward pass.
        
        Args:
            latents: Input latent tensor
            timestep: Current diffusion timestep
            is_cond: Whether this is a conditional (True) or unconditional (False) pass
            **kwargs: Additional arguments for the model
            
        Returns:
            Tensor: Model output
        """
        if not self.enable_teacache:
            # If TeaCache is disabled, just call the model normally
            return self.dit(latents, timestep=timestep, **kwargs)
        
        # TeaCache optimization logic
        if is_cond:
            # For conditional pass
            # Always calculate on first and last step
            if self.teacache_first_step or self.teacache_cnt == 0 or self.teacache_cnt == self.teacache_num_steps - 1:
                should_calc = True
                self.teacache_accumulated_rel_l1_distance = 0
                self.teacache_first_step = False  # Clear first step flag
            else:
                # Calculate relative L1 distance between current and previous input
                if self.teacache_previous_modulated_input is not None:
                    try:
                        # Calculate the relative L1 distance
                        abs_diff = (timestep - self.teacache_previous_modulated_input).abs().mean()
                        mean_prev = self.teacache_previous_modulated_input.abs().mean() + 1e-8
                        rel_l1 = (abs_diff / mean_prev).cpu().item()
                        
                        # Apply polynomial scaling to the distance
                        if self.teacache_coefficients is not None:
                            rescale_func = np.poly1d(self.teacache_coefficients)
                            scaled_distance = rescale_func(rel_l1)
                            
                            # For 14B model, ensure absolute value is used since the polynomial 
                            # can produce negative values that would otherwise prevent reaching threshold
                            if self.teacache_coefficients[0] < 0:  # 14B model has negative first coefficient
                                scaled_distance = abs(scaled_distance)
                                
                            self.teacache_accumulated_rel_l1_distance += scaled_distance
                        else:
                            # If no coefficients, use raw distance
                            self.teacache_accumulated_rel_l1_distance += rel_l1
                    except Exception as e:
                        print(f"Error in TeaCache calculation: {e}")
                        should_calc = True
                        self.teacache_accumulated_rel_l1_distance = 0
                
                # Decide whether to skip computation
                if self.teacache_accumulated_rel_l1_distance < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.teacache_accumulated_rel_l1_distance = 0
            
            # Store current modulated input for next comparison
            self.teacache_previous_modulated_input = timestep.detach().clone()
            
            # Update counter - only increment on conditional pass
            self.teacache_cnt = (self.teacache_cnt + 1) % self.teacache_num_steps
            
            # Store decision for unconditional pass
            self.teacache_should_calc = should_calc
        else:
            # For unconditional pass, use the same decision as the conditional pass
            should_calc = self.teacache_should_calc
        
        # Apply the optimization
        if not should_calc and self.teacache_previous_residual_cond is not None and self.teacache_previous_residual_uncond is not None:
            # Skip computation and reuse previous residual
            if is_cond:
                return latents + self.teacache_previous_residual_cond
            else:
                return latents + self.teacache_previous_residual_uncond
        else:
            # Perform normal computation
            ori_latents = latents.detach().clone()
            output = self.dit(latents, timestep=timestep, **kwargs)
            
            # Store the residual for future use
            residual = output - ori_latents
            if is_cond:
                self.teacache_previous_residual_cond = residual.detach().clone()
            else:
                self.teacache_previous_residual_uncond = residual.detach().clone()
                
            return output


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        # TeaCache parameters
        enable_teacache=False,
        teacache_thresh=0.05,
        teacache_model_size=None,  # Auto-detected if None
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Set up TeaCache
        self.setup_teacache(
            enable=enable_teacache, 
            thresh=teacache_thresh, 
            num_steps=num_inference_steps,
            model_size=teacache_model_size
        )
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32).to(self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=noise.dtype, device=noise.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # Reset TeaCache state explicitly at beginning of sampling
        if self.enable_teacache:
            self.teacache_cnt = 0
            self.teacache_accumulated_rel_l1_distance = 0
            self.teacache_previous_modulated_input = None
            self.teacache_previous_residual_cond = None
            self.teacache_previous_residual_uncond = None
            self.teacache_should_calc = True
            self.teacache_first_step = True

        # Denoise
        self.load_models_to_device(["dit"])
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)

                # Inference with TeaCache
                if self.enable_teacache:
                    # Apply TeaCache for conditional pass
                    noise_pred_posi = self.apply_teacache(
                        latents, 
                        timestep, 
                        is_cond=True, 
                        **prompt_emb_posi, 
                        **image_emb, 
                        **extra_input, 
                        cond_flag=True
                    )
                    
                    if cfg_scale != 1.0:
                        # Apply TeaCache for unconditional pass
                        noise_pred_nega = self.apply_teacache(
                            latents, 
                            timestep, 
                            is_cond=False, 
                            **prompt_emb_nega, 
                            **image_emb, 
                            **extra_input, 
                            cond_flag=False
                        )
                        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                    else:
                        noise_pred = noise_pred_posi
                else:
                    # Standard inference without TeaCache
                    noise_pred_posi = self.dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input)
                    if cfg_scale != 1.0:
                        noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_nega, **image_emb, **extra_input)
                        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                    else:
                        noise_pred = noise_pred_posi

                # Scheduler
                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
