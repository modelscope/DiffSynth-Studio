from ..models import ModelManager, SDTextEncoder, SDUNet, SDVAEDecoder, SDVAEEncoder, SDMotionModel
from ..models.sd_unet import PushBlock, PopBlock
from ..controlnets import MultiControlNetManager, ControlNetUnit, ControlNetConfigUnit, Annotator
from ..prompts import SDPrompter
from ..schedulers import EnhancedDDIMScheduler
from typing import List
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np


def lets_dance(
    unet: SDUNet,
    motion_modules: SDMotionModel = None,
    controlnet: MultiControlNetManager = None,
    sample = None,
    timestep = None,
    encoder_hidden_states = None,
    controlnet_frames = None,
    unet_batch_size = 1,
    controlnet_batch_size = 1,
    device = "cuda",
    vram_limit_level = 0,
):
    # 1. ControlNet
    #     This part will be repeated on overlapping frames if animatediff_batch_size > animatediff_stride.
    #     I leave it here because I intend to do something interesting on the ControlNets.
    controlnet_insert_block_id = 30
    if controlnet is not None and controlnet_frames is not None:
        res_stacks = []
        # process controlnet frames with batch
        for batch_id in range(0, sample.shape[0], controlnet_batch_size):
            batch_id_ = min(batch_id + controlnet_batch_size, sample.shape[0])
            res_stack = controlnet(
                sample[batch_id: batch_id_],
                timestep,
                encoder_hidden_states[batch_id: batch_id_],
                controlnet_frames[:, batch_id: batch_id_]
            )
            if vram_limit_level >= 1:
                res_stack = [res.cpu() for res in res_stack]
            res_stacks.append(res_stack)
        # concat the residual
        additional_res_stack = []
        for i in range(len(res_stacks[0])):
            res = torch.concat([res_stack[i] for res_stack in res_stacks], dim=0)
            additional_res_stack.append(res)
    else:
        additional_res_stack = None

    # 2. time
    time_emb = unet.time_proj(timestep[None]).to(sample.dtype)
    time_emb = unet.time_embedding(time_emb)

    # 3. pre-process
    hidden_states = unet.conv_in(sample)
    text_emb = encoder_hidden_states
    res_stack = [hidden_states.cpu() if vram_limit_level>=1 else hidden_states]

    # 4. blocks
    for block_id, block in enumerate(unet.blocks):
        # 4.1 UNet
        if isinstance(block, PushBlock):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)
            if vram_limit_level>=1:
                res_stack[-1] = res_stack[-1].cpu()
        elif isinstance(block, PopBlock):
            if vram_limit_level>=1:
                res_stack[-1] = res_stack[-1].to(device)
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)
        else:
            hidden_states_input = hidden_states
            hidden_states_output = []
            for batch_id in range(0, sample.shape[0], unet_batch_size):
                batch_id_ = min(batch_id + unet_batch_size, sample.shape[0])
                hidden_states, _, _, _ = block(hidden_states_input[batch_id: batch_id_], time_emb, text_emb[batch_id: batch_id_], res_stack)
                hidden_states_output.append(hidden_states)
            hidden_states = torch.concat(hidden_states_output, dim=0)
        # 4.2 AnimateDiff
        if motion_modules is not None:
            if block_id in motion_modules.call_block_id:
                motion_module_id = motion_modules.call_block_id[block_id]
                hidden_states, time_emb, text_emb, res_stack = motion_modules.motion_modules[motion_module_id](
                    hidden_states, time_emb, text_emb, res_stack,
                    batch_size=1
                )
        # 4.3 ControlNet
        if block_id == controlnet_insert_block_id and additional_res_stack is not None:
            hidden_states += additional_res_stack.pop().to(device)
            if vram_limit_level>=1:
                res_stack = [(res.to(device) + additional_res.to(device)).cpu() for res, additional_res in zip(res_stack, additional_res_stack)]
            else:
                res_stack = [res + additional_res for res, additional_res in zip(res_stack, additional_res_stack)]
    
    # 5. output
    hidden_states = unet.conv_norm_out(hidden_states)
    hidden_states = unet.conv_act(hidden_states)
    hidden_states = unet.conv_out(hidden_states)

    return hidden_states


def lets_dance_with_long_video(
    unet: SDUNet,
    motion_modules: SDMotionModel = None,
    controlnet: MultiControlNetManager = None,
    sample = None,
    timestep = None,
    encoder_hidden_states = None,
    controlnet_frames = None,
    unet_batch_size = 1,
    controlnet_batch_size = 1,
    animatediff_batch_size = 16,
    animatediff_stride = 8,
    device = "cuda",
    vram_limit_level = 0,
):
    num_frames = sample.shape[0]
    hidden_states_output = [(torch.zeros(sample[0].shape, dtype=sample[0].dtype), 0) for i in range(num_frames)]

    for batch_id in range(0, num_frames, animatediff_stride):
        batch_id_ = min(batch_id + animatediff_batch_size, num_frames)

        # process this batch
        hidden_states_batch = lets_dance(
            unet, motion_modules, controlnet,
            sample[batch_id: batch_id_].to(device),
            timestep,
            encoder_hidden_states[batch_id: batch_id_].to(device),
            controlnet_frames[:, batch_id: batch_id_].to(device) if controlnet_frames is not None else None,
            unet_batch_size=unet_batch_size, controlnet_batch_size=controlnet_batch_size, device=device, vram_limit_level=vram_limit_level
        ).cpu()

        # update hidden_states
        for i, hidden_states_updated in zip(range(batch_id, batch_id_), hidden_states_batch):
            bias = max(1 - abs(i - (batch_id + batch_id_ - 1) / 2) / ((batch_id_ - batch_id - 1) / 2), 1e-2)
            hidden_states, num = hidden_states_output[i]
            hidden_states = hidden_states * (num / (num + bias)) + hidden_states_updated * (bias / (num + bias))
            hidden_states_output[i] = (hidden_states, num + 1)

    # output
    hidden_states = torch.stack([h for h, _ in hidden_states_output])
    return hidden_states


class SDVideoPipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16, use_animatediff=True):
        super().__init__()
        self.scheduler = EnhancedDDIMScheduler(beta_schedule="linear" if use_animatediff else "scaled_linear")
        self.prompter = SDPrompter()
        self.device = device
        self.torch_dtype = torch_dtype
        # models
        self.text_encoder: SDTextEncoder = None
        self.unet: SDUNet = None
        self.vae_decoder: SDVAEDecoder = None
        self.vae_encoder: SDVAEEncoder = None
        self.controlnet: MultiControlNetManager = None
        self.motion_modules: SDMotionModel = None


    def fetch_main_models(self, model_manager: ModelManager):
        self.text_encoder = model_manager.text_encoder
        self.unet = model_manager.unet
        self.vae_decoder = model_manager.vae_decoder
        self.vae_encoder = model_manager.vae_encoder
        # load textual inversion
        self.prompter.load_textual_inversion(model_manager.textual_inversion_dict)


    def fetch_controlnet_models(self, model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[]):
        controlnet_units = []
        for config in controlnet_config_units:
            controlnet_unit = ControlNetUnit(
                Annotator(config.processor_id),
                model_manager.get_model_with_model_path(config.model_path),
                config.scale
            )
            controlnet_units.append(controlnet_unit)
        self.controlnet = MultiControlNetManager(controlnet_units)


    def fetch_motion_modules(self, model_manager: ModelManager):
        if "motion_modules" in model_manager.model:
            self.motion_modules = model_manager.motion_modules


    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[]):
        pipe = SDVideoPipeline(
            device=model_manager.device,
            torch_dtype=model_manager.torch_dtype,
            use_animatediff="motion_modules" in model_manager.model
        )
        pipe.fetch_main_models(model_manager)
        pipe.fetch_motion_modules(model_manager)
        pipe.fetch_controlnet_models(model_manager, controlnet_config_units)
        return pipe
    

    def preprocess_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
        return image
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        image = image.cpu().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image
    

    def decode_images(self, latents, tiled=False, tile_size=64, tile_stride=32):
        images = [
            self.decode_image(latents[frame_id: frame_id+1], tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            for frame_id in range(latents.shape[0])
        ]
        return images
    

    def encode_images(self, processed_images, tiled=False, tile_size=64, tile_stride=32):
        latents = []
        for image in processed_images:
            image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)
            latent = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).cpu()
            latents.append(latent)
        latents = torch.concat(latents, dim=0)
        return latents
    

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        cfg_scale=7.5,
        clip_skip=1,
        num_frames=None,
        input_frames=None,
        controlnet_frames=None,
        denoising_strength=1.0,
        height=512,
        width=512,
        num_inference_steps=20,
        vram_limit_level=0,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Encode prompts
        prompt_emb_posi = self.prompter.encode_prompt(self.text_encoder, prompt, clip_skip=clip_skip, device=self.device).cpu()
        prompt_emb_nega = self.prompter.encode_prompt(self.text_encoder, negative_prompt, clip_skip=clip_skip, device=self.device).cpu()
        prompt_emb_posi = prompt_emb_posi.repeat(num_frames, 1, 1)
        prompt_emb_nega = prompt_emb_nega.repeat(num_frames, 1, 1)

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        noise = torch.randn((num_frames, 4, height//8, width//8), device="cpu", dtype=self.torch_dtype)
        if input_frames is None:
            latents = noise
        else:
            latents = self.encode_images(input_frames)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])

        # Prepare ControlNets
        if controlnet_frames is not None:
            controlnet_frames = torch.stack([
                self.controlnet.process_image(controlnet_frame).to(self.torch_dtype)
                for controlnet_frame in progress_bar_cmd(controlnet_frames)
            ], dim=1)
        
        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.IntTensor((timestep,))[0].to(self.device)

            # Classifier-free guidance
            noise_pred_posi = lets_dance_with_long_video(
                self.unet, motion_modules=self.motion_modules, controlnet=self.controlnet,
                sample=latents, timestep=timestep, encoder_hidden_states=prompt_emb_posi, controlnet_frames=controlnet_frames,
                device=self.device, vram_limit_level=vram_limit_level
            )
            noise_pred_nega = lets_dance_with_long_video(
                self.unet, motion_modules=self.motion_modules, controlnet=self.controlnet,
                sample=latents, timestep=timestep, encoder_hidden_states=prompt_emb_nega, controlnet_frames=controlnet_frames,
                device=self.device, vram_limit_level=vram_limit_level
            )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

            # DDIM
            latents = self.scheduler.step(noise_pred, timestep, latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        output_frames = self.decode_images(latents)

        return output_frames
