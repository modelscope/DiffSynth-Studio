from ..models import ModelManager, SD3TextEncoder1, HunyuanVideoVAEDecoder, HunyuanVideoVAEEncoder
from ..models.hunyuan_video_dit import HunyuanVideoDiT
from ..models.hunyuan_video_text_encoder import HunyuanVideoLLMEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import HunyuanVideoPrompter
import torch
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm



class HunyuanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=7.0, sigma_min=0.0, extra_one_step=True)
        self.prompter = HunyuanVideoPrompter()
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: HunyuanVideoLLMEncoder = None
        self.dit: HunyuanVideoDiT = None
        self.vae_decoder: HunyuanVideoVAEDecoder = None
        self.vae_encoder: HunyuanVideoVAEEncoder = None
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'dit', 'vae_decoder', 'vae_encoder']
        self.vram_management = False


    def enable_vram_management(self):
        self.vram_management = True
        self.enable_cpu_offload()
        self.text_encoder_2.enable_auto_offload(dtype=self.torch_dtype, device=self.device)
        self.dit.enable_auto_offload(dtype=self.torch_dtype, device=self.device)


    def fetch_models(self, model_manager: ModelManager):
        self.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("hunyuan_video_text_encoder_2")
        self.dit = model_manager.fetch_model("hunyuan_video_dit")
        self.vae_decoder = model_manager.fetch_model("hunyuan_video_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("hunyuan_video_vae_encoder")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, enable_vram_management=True):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = HunyuanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if enable_vram_management:
            pipe.enable_vram_management()
        return pipe


    def encode_prompt(self, prompt, positive=True, clip_sequence_length=77, llm_sequence_length=256):
        prompt_emb, pooled_prompt_emb, text_mask = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive, clip_sequence_length=clip_sequence_length, llm_sequence_length=llm_sequence_length
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_mask": text_mask}


    def prepare_extra_input(self, latents=None, guidance=1.0):
        freqs_cos, freqs_sin = self.dit.prepare_freqs(latents)
        guidance = torch.Tensor([guidance] * latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
        return {"freqs_cos": freqs_cos, "freqs_sin": freqs_sin, "guidance": guidance}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames


    def encode_video(self, frames, tile_size=(17, 30, 30), tile_stride=(12, 20, 20)):
        tile_size = ((tile_size[0] - 1) * 4 + 1, tile_size[1] * 8, tile_size[2] * 8)
        tile_stride = (tile_stride[0] * 4, tile_stride[1] * 8, tile_stride[2] * 8)
        latents = self.vae_encoder.encode_video(frames, tile_size=tile_size, tile_stride=tile_stride)
        return latents


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device=None,
        height=720,
        width=1280,
        num_frames=129,
        embedded_guidance=6.0,
        cfg_scale=1.0,
        num_inference_steps=30,
        tea_cache_l1_thresh=None,
        tile_size=(17, 30, 30),
        tile_stride=(12, 20, 20),
        step_processor=None,
        progress_bar_cmd=lambda x: x,
        progress_bar_st=None,
    ):
        # Tiler parameters
        tiler_kwargs = {"tile_size": tile_size, "tile_stride": tile_stride}
        
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Initialize noise
        rand_device = self.device if rand_device is None else rand_device
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=self.torch_dtype).to(self.device)
        if input_video is not None:
            self.load_models_to_device(['vae_encoder'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        
        # Encode prompts
        self.load_models_to_device(["text_encoder_1"] if self.vram_management else ["text_encoder_1", "text_encoder_2"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # TeaCache
        tea_cache_kwargs = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None}

        # Denoise
        self.load_models_to_device([] if self.vram_management else ["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)
            print(f"Step {progress_id + 1} / {len(self.scheduler.timesteps)}")

            # Inference
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                noise_pred_posi = lets_dance_hunyuan_video(self.dit, latents, timestep, **prompt_emb_posi, **extra_input, **tea_cache_kwargs)
                if cfg_scale != 1.0:
                    noise_pred_nega = lets_dance_hunyuan_video(self.dit, latents, timestep, **prompt_emb_nega, **extra_input)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

            # (Experimental feature, may be removed in the future)
            if step_processor is not None:
                self.load_models_to_device(['vae_decoder'])
                rendered_frames = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents, to_final=True)
                rendered_frames = self.vae_decoder.decode_video(rendered_frames, **tiler_kwargs)
                rendered_frames = self.tensor2video(rendered_frames[0])
                rendered_frames = step_processor(rendered_frames, original_frames=input_video)
                self.load_models_to_device(['vae_encoder'])
                rendered_frames = self.preprocess_images(rendered_frames)
                rendered_frames = torch.stack(rendered_frames, dim=2)
                target_latents = self.encode_video(rendered_frames).to(dtype=self.torch_dtype, device=self.device)
                noise_pred = self.scheduler.return_to_timestep(self.scheduler.timesteps[progress_id], latents, target_latents)
                self.load_models_to_device([] if self.vram_management else ["dit"])

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        # Decode
        self.load_models_to_device(['vae_decoder'])
        frames = self.vae_decoder.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

    def check(self, dit: HunyuanVideoDiT, img, vec):
        img_ = img.clone()
        vec_ = vec.clone()
        img_mod1_shift, img_mod1_scale, _, _, _, _ = dit.double_blocks[0].component_a.mod(vec_).chunk(6, dim=-1)
        normed_inp = dit.double_blocks[0].component_a.norm1(img_)
        modulated_inp = normed_inp * (1 + img_mod1_scale.unsqueeze(1)) + img_mod1_shift.unsqueeze(1)
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else: 
            coefficients = [7.33226126e+02, -4.01131952e+02,  6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp 
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = img.clone()
        return not should_calc
    
    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def lets_dance_hunyuan_video(
    dit: HunyuanVideoDiT,
    x: torch.Tensor,
    t: torch.Tensor,
    prompt_emb: torch.Tensor = None,
    text_mask: torch.Tensor = None,
    pooled_prompt_emb: torch.Tensor = None,
    freqs_cos: torch.Tensor = None,
    freqs_sin: torch.Tensor = None,
    guidance: torch.Tensor = None,
    tea_cache: TeaCache = None,
    **kwargs
):
    B, C, T, H, W = x.shape

    vec = dit.time_in(t, dtype=torch.float32) + dit.vector_in(pooled_prompt_emb) + dit.guidance_in(guidance * 1000, dtype=torch.float32)
    img = dit.img_in(x)
    txt = dit.txt_in(prompt_emb, t, text_mask)

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, img, vec)
    else:
        tea_cache_update = False

    if tea_cache_update:
        print("TeaCache skip forward.")
        img = tea_cache.update(img)
    else:
        for block in tqdm(dit.double_blocks, desc="Double stream blocks"):
            img, txt = block(img, txt, vec, (freqs_cos, freqs_sin))
        
        x = torch.concat([img, txt], dim=1)
        for block in tqdm(dit.single_blocks, desc="Single stream blocks"):
            x = block(x, vec, (freqs_cos, freqs_sin))
        img = x[:, :-256]

        if tea_cache is not None:
            tea_cache.store(img)
    img = dit.final_layer(img, vec)
    img = dit.unpatchify(img, T=T//1, H=H//2, W=W//2)
    return img
