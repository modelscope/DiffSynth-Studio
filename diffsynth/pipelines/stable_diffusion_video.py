from ..models import ModelManager, SDTextEncoder, SDUNet, SDVAEDecoder, SDVAEEncoder, SDMotionModel
from ..controlnets import MultiControlNetManager, ControlNetUnit, ControlNetConfigUnit, Annotator
from ..prompts import SDPrompter
from ..schedulers import EnhancedDDIMScheduler
from ..data import VideoData, save_frames, save_video
from .dancer import lets_dance
from ..processors.sequencial_processor import SequencialProcessor
from typing import List
import torch, os, json
from tqdm import tqdm
from PIL import Image
import numpy as np


def lets_dance_with_long_video(
    unet: SDUNet,
    motion_modules: SDMotionModel = None,
    controlnet: MultiControlNetManager = None,
    sample = None,
    timestep = None,
    encoder_hidden_states = None,
    controlnet_frames = None,
    animatediff_batch_size = 16,
    animatediff_stride = 8,
    unet_batch_size = 1,
    controlnet_batch_size = 1,
    cross_frame_attention = False,
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
            controlnet_frames=controlnet_frames[:, batch_id: batch_id_].to(device) if controlnet_frames is not None else None,
            unet_batch_size=unet_batch_size, controlnet_batch_size=controlnet_batch_size,
            cross_frame_attention=cross_frame_attention,
            device=device, vram_limit_level=vram_limit_level
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


    def fetch_prompter(self, model_manager: ModelManager):
        self.prompter.load_from_model_manager(model_manager)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[]):
        pipe = SDVideoPipeline(
            device=model_manager.device,
            torch_dtype=model_manager.torch_dtype,
            use_animatediff="motion_modules" in model_manager.model
        )
        pipe.fetch_main_models(model_manager)
        pipe.fetch_motion_modules(model_manager)
        pipe.fetch_prompter(model_manager)
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
        animatediff_batch_size = 16,
        animatediff_stride = 8,
        unet_batch_size = 1,
        controlnet_batch_size = 1,
        cross_frame_attention = False,
        smoother=None,
        smoother_progress_ids=[],
        vram_limit_level=0,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if self.motion_modules is None:
            noise = torch.randn((1, 4, height//8, width//8), device="cpu", dtype=self.torch_dtype).repeat(num_frames, 1, 1, 1)
        else:
            noise = torch.randn((num_frames, 4, height//8, width//8), device="cpu", dtype=self.torch_dtype)
        if input_frames is None or denoising_strength == 1.0:
            latents = noise
        else:
            latents = self.encode_images(input_frames)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])

        # Encode prompts
        prompt_emb_posi = self.prompter.encode_prompt(self.text_encoder, prompt, clip_skip=clip_skip, device=self.device, positive=True).cpu()
        prompt_emb_nega = self.prompter.encode_prompt(self.text_encoder, negative_prompt, clip_skip=clip_skip, device=self.device, positive=False).cpu()
        prompt_emb_posi = prompt_emb_posi.repeat(num_frames, 1, 1)
        prompt_emb_nega = prompt_emb_nega.repeat(num_frames, 1, 1)

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
        
        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.IntTensor((timestep,))[0].to(self.device)

            # Classifier-free guidance
            noise_pred_posi = lets_dance_with_long_video(
                self.unet, motion_modules=self.motion_modules, controlnet=self.controlnet,
                sample=latents, timestep=timestep, encoder_hidden_states=prompt_emb_posi, controlnet_frames=controlnet_frames,
                animatediff_batch_size=animatediff_batch_size, animatediff_stride=animatediff_stride,
                unet_batch_size=unet_batch_size, controlnet_batch_size=controlnet_batch_size,
                cross_frame_attention=cross_frame_attention,
                device=self.device, vram_limit_level=vram_limit_level
            )
            noise_pred_nega = lets_dance_with_long_video(
                self.unet, motion_modules=self.motion_modules, controlnet=self.controlnet,
                sample=latents, timestep=timestep, encoder_hidden_states=prompt_emb_nega, controlnet_frames=controlnet_frames,
                animatediff_batch_size=animatediff_batch_size, animatediff_stride=animatediff_stride,
                unet_batch_size=unet_batch_size, controlnet_batch_size=controlnet_batch_size,
                cross_frame_attention=cross_frame_attention,
                device=self.device, vram_limit_level=vram_limit_level
            )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

            # DDIM and smoother
            if smoother is not None and progress_id in smoother_progress_ids:
                rendered_frames = self.scheduler.step(noise_pred, timestep, latents, to_final=True)
                rendered_frames = self.decode_images(rendered_frames)
                rendered_frames = smoother(rendered_frames, original_frames=input_frames)
                target_latents = self.encode_images(rendered_frames)
                noise_pred = self.scheduler.return_to_timestep(timestep, latents, target_latents)
            latents = self.scheduler.step(noise_pred, timestep, latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        output_frames = self.decode_images(latents)

        # Post-process
        if smoother is not None and (num_inference_steps in smoother_progress_ids or -1 in smoother_progress_ids):
            output_frames = smoother(output_frames, original_frames=input_frames)

        return output_frames



class SDVideoPipelineRunner:
    def __init__(self, in_streamlit=False):
        self.in_streamlit = in_streamlit


    def load_pipeline(self, model_list, textual_inversion_folder, device, lora_alphas, controlnet_units):
        # Load models
        model_manager = ModelManager(torch_dtype=torch.float16, device=device)
        model_manager.load_textual_inversions(textual_inversion_folder)
        model_manager.load_models(model_list, lora_alphas=lora_alphas)
        pipe = SDVideoPipeline.from_model_manager(
            model_manager,
            [
                ControlNetConfigUnit(
                    processor_id=unit["processor_id"],
                    model_path=unit["model_path"],
                    scale=unit["scale"]
                ) for unit in controlnet_units
            ]
        )
        return model_manager, pipe
    

    def load_smoother(self, model_manager, smoother_configs):
        smoother = SequencialProcessor.from_model_manager(model_manager, smoother_configs)
        return smoother


    def synthesize_video(self, model_manager, pipe, seed, smoother, **pipeline_inputs):
        torch.manual_seed(seed)
        if self.in_streamlit:
            import streamlit as st
            progress_bar_st = st.progress(0.0)
            output_video = pipe(**pipeline_inputs, smoother=smoother, progress_bar_st=progress_bar_st)
            progress_bar_st.progress(1.0)
        else:
            output_video = pipe(**pipeline_inputs, smoother=smoother)
        model_manager.to("cpu")
        return output_video


    def load_video(self, video_file, image_folder, height, width, start_frame_id, end_frame_id):
        video = VideoData(video_file=video_file, image_folder=image_folder, height=height, width=width)
        if start_frame_id is None:
            start_frame_id = 0
        if end_frame_id is None:
            end_frame_id = len(video)
        frames = [video[i] for i in range(start_frame_id, end_frame_id)]
        return frames


    def add_data_to_pipeline_inputs(self, data, pipeline_inputs):
        pipeline_inputs["input_frames"] = self.load_video(**data["input_frames"])
        pipeline_inputs["num_frames"] = len(pipeline_inputs["input_frames"])
        pipeline_inputs["width"], pipeline_inputs["height"] = pipeline_inputs["input_frames"][0].size
        if len(data["controlnet_frames"]) > 0:
            pipeline_inputs["controlnet_frames"] = [self.load_video(**unit) for unit in data["controlnet_frames"]]
        return pipeline_inputs


    def save_output(self, video, output_folder, fps, config):
        os.makedirs(output_folder, exist_ok=True)
        save_frames(video, os.path.join(output_folder, "frames"))
        save_video(video, os.path.join(output_folder, "video.mp4"), fps=fps)
        config["pipeline"]["pipeline_inputs"]["input_frames"] = []
        config["pipeline"]["pipeline_inputs"]["controlnet_frames"] = []
        with open(os.path.join(output_folder, "config.json"), 'w') as file:
            json.dump(config, file, indent=4)


    def run(self, config):
        if self.in_streamlit:
            import streamlit as st
        if self.in_streamlit: st.markdown("Loading videos ...")
        config["pipeline"]["pipeline_inputs"] = self.add_data_to_pipeline_inputs(config["data"], config["pipeline"]["pipeline_inputs"])
        if self.in_streamlit: st.markdown("Loading videos ... done!")
        if self.in_streamlit: st.markdown("Loading models ...")
        model_manager, pipe = self.load_pipeline(**config["models"])
        if self.in_streamlit: st.markdown("Loading models ... done!")
        if "smoother_configs" in config:
            if self.in_streamlit: st.markdown("Loading smoother ...")
            smoother = self.load_smoother(model_manager, config["smoother_configs"])
            if self.in_streamlit: st.markdown("Loading smoother ... done!")
        else:
            smoother = None
        if self.in_streamlit: st.markdown("Synthesizing videos ...")
        output_video = self.synthesize_video(model_manager, pipe, config["pipeline"]["seed"], smoother, **config["pipeline"]["pipeline_inputs"])
        if self.in_streamlit: st.markdown("Synthesizing videos ... done!")
        if self.in_streamlit: st.markdown("Saving videos ...")
        self.save_output(output_video, config["data"]["output_folder"], config["data"]["fps"], config)
        if self.in_streamlit: st.markdown("Saving videos ... done!")
        if self.in_streamlit: st.markdown("Finished!")
        video_file = open(os.path.join(os.path.join(config["data"]["output_folder"], "video.mp4")), 'rb')
        if self.in_streamlit: st.video(video_file.read())
