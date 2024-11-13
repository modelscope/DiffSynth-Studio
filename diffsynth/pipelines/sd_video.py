from ..models import SDTextEncoder, SDUNet, SDVAEDecoder, SDVAEEncoder, SDIpAdapter, IpAdapterCLIPImageEmbedder, SDMotionModel
from ..models.model_manager import ModelManager
from ..controlnets import MultiControlNetManager, ControlNetUnit, ControlNetConfigUnit, Annotator
from ..prompters import SDPrompter
from ..schedulers import EnhancedDDIMScheduler
from .sd_image import SDImagePipeline
from .dancer import lets_dance
from typing import List
import torch
from tqdm import tqdm



def lets_dance_with_long_video(
    unet: SDUNet,
    motion_modules: SDMotionModel = None,
    controlnet: MultiControlNetManager = None,
    sample = None,
    timestep = None,
    encoder_hidden_states = None,
    ipadapter_kwargs_list = {},
    controlnet_frames = None,
    unet_batch_size = 1,
    controlnet_batch_size = 1,
    cross_frame_attention = False,
    tiled=False,
    tile_size=64,
    tile_stride=32,
    device="cuda",
    animatediff_batch_size=16,
    animatediff_stride=8,
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
            encoder_hidden_states,
            ipadapter_kwargs_list=ipadapter_kwargs_list,
            controlnet_frames=controlnet_frames[:, batch_id: batch_id_].to(device) if controlnet_frames is not None else None,
            unet_batch_size=unet_batch_size, controlnet_batch_size=controlnet_batch_size,
            cross_frame_attention=cross_frame_attention,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, device=device
        ).cpu()

        # update hidden_states
        for i, hidden_states_updated in zip(range(batch_id, batch_id_), hidden_states_batch):
            bias = max(1 - abs(i - (batch_id + batch_id_ - 1) / 2) / ((batch_id_ - batch_id - 1 + 1e-2) / 2), 1e-2)
            hidden_states, num = hidden_states_output[i]
            hidden_states = hidden_states * (num / (num + bias)) + hidden_states_updated * (bias / (num + bias))
            hidden_states_output[i] = (hidden_states, num + bias)

        if batch_id_ == num_frames:
            break

    # output
    hidden_states = torch.stack([h for h, _ in hidden_states_output])
    return hidden_states



class SDVideoPipeline(SDImagePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, use_original_animatediff=True):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = EnhancedDDIMScheduler(beta_schedule="linear" if use_original_animatediff else "scaled_linear")
        self.prompter = SDPrompter()
        # models
        self.text_encoder: SDTextEncoder = None
        self.unet: SDUNet = None
        self.vae_decoder: SDVAEDecoder = None
        self.vae_encoder: SDVAEEncoder = None
        self.controlnet: MultiControlNetManager = None
        self.ipadapter_image_encoder: IpAdapterCLIPImageEmbedder = None
        self.ipadapter: SDIpAdapter = None
        self.motion_modules: SDMotionModel = None


    def fetch_models(self, model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[]):
        # Main models
        self.text_encoder = model_manager.fetch_model("sd_text_encoder")
        self.unet = model_manager.fetch_model("sd_unet")
        self.vae_decoder = model_manager.fetch_model("sd_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("sd_vae_encoder")
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.load_prompt_refiners(model_manager, prompt_refiner_classes)

        # ControlNets
        controlnet_units = []
        for config in controlnet_config_units:
            controlnet_unit = ControlNetUnit(
                Annotator(config.processor_id, device=self.device),
                model_manager.fetch_model("sd_controlnet", config.model_path),
                config.scale
            )
            controlnet_units.append(controlnet_unit)
        self.controlnet = MultiControlNetManager(controlnet_units)

        # IP-Adapters
        self.ipadapter = model_manager.fetch_model("sd_ipadapter")
        self.ipadapter_image_encoder = model_manager.fetch_model("sd_ipadapter_clip_image_encoder")

        # Motion Modules
        self.motion_modules = model_manager.fetch_model("sd_motion_modules")
        if self.motion_modules is None:
            self.scheduler = EnhancedDDIMScheduler(beta_schedule="scaled_linear")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[]):
        pipe = SDVideoPipeline(
            device=model_manager.device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_models(model_manager, controlnet_config_units, prompt_refiner_classes)
        return pipe
    

    def decode_video(self, latents, tiled=False, tile_size=64, tile_stride=32):
        images = [
            self.decode_image(latents[frame_id: frame_id+1], tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            for frame_id in range(latents.shape[0])
        ]
        return images
    

    def encode_video(self, processed_images, tiled=False, tile_size=64, tile_stride=32):
        latents = []
        for image in processed_images:
            image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)
            latent = self.encode_image(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            latents.append(latent.cpu())
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
        ipadapter_images=None,
        ipadapter_scale=1.0,
        controlnet_frames=None,
        denoising_strength=1.0,
        height=512,
        width=512,
        num_inference_steps=20,
        animatediff_batch_size = 16,
        animatediff_stride = 8,
        unet_batch_size = 1,
        controlnet_batch_size = 1,
        cross_frame_attention = False,
        smoother=None,
        smoother_progress_ids=[],
        tiled=False,
        tile_size=64,
        tile_stride=32,
        seed=None,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        height, width = self.check_resize_height_width(height, width)
        
        # Tiler parameters, batch size ...
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        other_kwargs = {
            "animatediff_batch_size": animatediff_batch_size, "animatediff_stride": animatediff_stride,
            "unet_batch_size": unet_batch_size, "controlnet_batch_size": controlnet_batch_size,
            "cross_frame_attention": cross_frame_attention,
        }

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if self.motion_modules is None:
            noise = self.generate_noise((1, 4, height//8, width//8), seed=seed, device="cpu", dtype=self.torch_dtype).repeat(num_frames, 1, 1, 1)
        else:
            noise = self.generate_noise((num_frames, 4, height//8, width//8), seed=seed, device="cpu", dtype=self.torch_dtype)
        if input_frames is None or denoising_strength == 1.0:
            latents = noise
        else:
            latents = self.encode_video(input_frames, **tiler_kwargs)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])

        # Encode prompts
        prompt_emb_posi = self.encode_prompt(prompt, clip_skip=clip_skip, positive=True)
        prompt_emb_nega = self.encode_prompt(negative_prompt, clip_skip=clip_skip, positive=False)

        # IP-Adapter
        if ipadapter_images is not None:
            ipadapter_image_encoding = self.ipadapter_image_encoder(ipadapter_images)
            ipadapter_kwargs_list_posi = {"ipadapter_kwargs_list": self.ipadapter(ipadapter_image_encoding, scale=ipadapter_scale)}
            ipadapter_kwargs_list_nega = {"ipadapter_kwargs_list": self.ipadapter(torch.zeros_like(ipadapter_image_encoding))}
        else:
            ipadapter_kwargs_list_posi, ipadapter_kwargs_list_nega = {"ipadapter_kwargs_list": {}}, {"ipadapter_kwargs_list": {}}

        # Prepare ControlNets
        if controlnet_frames is not None:
            if isinstance(controlnet_frames[0], list):
                controlnet_frames_ = []
                for processor_id in range(len(controlnet_frames)):
                    controlnet_frames_.append(
                        torch.stack([
                            self.controlnet.process_image(controlnet_frame, processor_id=processor_id).to(self.torch_dtype)
                            for controlnet_frame in progress_bar_cmd(controlnet_frames[processor_id])
                        ], dim=1)
                    )
                controlnet_frames = torch.concat(controlnet_frames_, dim=0)
            else:
                controlnet_frames = torch.stack([
                    self.controlnet.process_image(controlnet_frame).to(self.torch_dtype)
                    for controlnet_frame in progress_bar_cmd(controlnet_frames)
                ], dim=1)
            controlnet_kwargs = {"controlnet_frames": controlnet_frames}
        else:
            controlnet_kwargs = {"controlnet_frames": None}
        
        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            noise_pred_posi = lets_dance_with_long_video(
                self.unet, motion_modules=self.motion_modules, controlnet=self.controlnet,
                sample=latents, timestep=timestep,
                **prompt_emb_posi, **controlnet_kwargs, **ipadapter_kwargs_list_posi, **other_kwargs, **tiler_kwargs,
                device=self.device,
            )
            noise_pred_nega = lets_dance_with_long_video(
                self.unet, motion_modules=self.motion_modules, controlnet=self.controlnet,
                sample=latents, timestep=timestep,
                **prompt_emb_nega, **controlnet_kwargs, **ipadapter_kwargs_list_nega, **other_kwargs, **tiler_kwargs,
                device=self.device,
            )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

            # DDIM and smoother
            if smoother is not None and progress_id in smoother_progress_ids:
                rendered_frames = self.scheduler.step(noise_pred, timestep, latents, to_final=True)
                rendered_frames = self.decode_video(rendered_frames)
                rendered_frames = smoother(rendered_frames, original_frames=input_frames)
                target_latents = self.encode_video(rendered_frames)
                noise_pred = self.scheduler.return_to_timestep(timestep, latents, target_latents)
            latents = self.scheduler.step(noise_pred, timestep, latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        output_frames = self.decode_video(latents, **tiler_kwargs)

        # Post-process
        if smoother is not None and (num_inference_steps in smoother_progress_ids or -1 in smoother_progress_ids):
            output_frames = smoother(output_frames, original_frames=input_frames)

        return output_frames
