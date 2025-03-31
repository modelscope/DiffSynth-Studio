from ..models import ModelManager, FluxDiT, SD3TextEncoder1, FluxTextEncoder2, FluxVAEDecoder, FluxVAEEncoder, FluxIpAdapter
from ..controlnets import FluxMultiControlNetManager, ControlNetUnit, ControlNetConfigUnit, Annotator
from ..prompters import FluxPrompter
from ..schedulers import FlowMatchScheduler
from .base import BasePipeline
from typing import List
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from ..models.tiler import FastTileWorker
from transformers import SiglipVisionModel
from copy import deepcopy
from transformers.models.t5.modeling_t5 import T5LayerNorm, T5DenseActDense, T5DenseGatedActDense
from ..models.flux_dit import RMSNorm
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear


class FluxImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=16, width_division_factor=16)
        self.scheduler = FlowMatchScheduler()
        self.prompter = FluxPrompter()
        # models
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: FluxTextEncoder2 = None
        self.dit: FluxDiT = None
        self.vae_decoder: FluxVAEDecoder = None
        self.vae_encoder: FluxVAEEncoder = None
        self.controlnet: FluxMultiControlNetManager = None
        self.ipadapter: FluxIpAdapter = None
        self.ipadapter_image_encoder: SiglipVisionModel = None
        self.infinityou_processor: InfinitYou = None
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'dit', 'vae_decoder', 'vae_encoder', 'controlnet', 'ipadapter', 'ipadapter_image_encoder']


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder_1.parameters())).dtype
        enable_vram_management(
            self.text_encoder_1,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
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
        dtype = next(iter(self.text_encoder_2.parameters())).dtype
        enable_vram_management(
            self.text_encoder_2,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
                T5DenseActDense: AutoWrappedModule,
                T5DenseGatedActDense: AutoWrappedModule,
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
                RMSNorm: AutoWrappedModule,
                torch.nn.Linear: AutoWrappedLinear,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cuda",
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
        dtype = next(iter(self.vae_decoder.parameters())).dtype
        enable_vram_management(
            self.vae_decoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.GroupNorm: AutoWrappedModule,
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
        dtype = next(iter(self.vae_encoder.parameters())).dtype
        enable_vram_management(
            self.vae_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.GroupNorm: AutoWrappedModule,
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


    def denoising_model(self):
        return self.dit


    def fetch_models(self, model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[], prompt_extender_classes=[]):
        self.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("flux_text_encoder_2")
        self.dit = model_manager.fetch_model("flux_dit")
        self.vae_decoder = model_manager.fetch_model("flux_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("flux_vae_encoder")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)
        self.prompter.load_prompt_refiners(model_manager, prompt_refiner_classes)
        self.prompter.load_prompt_extenders(model_manager, prompt_extender_classes)

        # ControlNets
        controlnet_units = []
        for config in controlnet_config_units:
            controlnet_unit = ControlNetUnit(
                Annotator(config.processor_id, device=self.device, skip_processor=config.skip_processor),
                model_manager.fetch_model("flux_controlnet", config.model_path),
                config.scale
            )
            controlnet_units.append(controlnet_unit)
        self.controlnet = FluxMultiControlNetManager(controlnet_units)

        # IP-Adapters
        self.ipadapter = model_manager.fetch_model("flux_ipadapter")
        self.ipadapter_image_encoder = model_manager.fetch_model("siglip_vision_model")

        # InfiniteYou
        self.image_proj_model = model_manager.fetch_model("infiniteyou_image_projector")
        if self.image_proj_model is not None:
            self.infinityou_processor = InfinitYou(device=self.device)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[], prompt_extender_classes=[], device=None, torch_dtype=None):
        pipe = FluxImagePipeline(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype if torch_dtype is None else torch_dtype,
        )
        pipe.fetch_models(model_manager, controlnet_config_units, prompt_refiner_classes, prompt_extender_classes)
        return pipe
    

    def encode_image(self, image, tiled=False, tile_size=64, tile_stride=32):
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        return image
    

    def encode_prompt(self, prompt, positive=True, t5_sequence_length=512):
        prompt_emb, pooled_prompt_emb, text_ids = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive, t5_sequence_length=t5_sequence_length
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}
    

    def prepare_extra_input(self, latents=None, guidance=1.0):
        latent_image_ids = self.dit.prepare_image_ids(latents)
        guidance = torch.Tensor([guidance] * latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
        return {"image_ids": latent_image_ids, "guidance": guidance}
    

    def apply_controlnet_mask_on_latents(self, latents, mask):
        mask = (self.preprocess_image(mask) + 1) / 2
        mask = mask.mean(dim=1, keepdim=True)
        mask = mask.to(dtype=self.torch_dtype, device=self.device)
        mask = 1 - torch.nn.functional.interpolate(mask, size=latents.shape[-2:])
        latents = torch.concat([latents, mask], dim=1)
        return latents
    

    def apply_controlnet_mask_on_image(self, image, mask):
        mask = mask.resize(image.size)
        mask = self.preprocess_image(mask).mean(dim=[0, 1])
        image = np.array(image)
        image[mask > 0] = 0
        image = Image.fromarray(image)
        return image
    

    def prepare_controlnet_input(self, controlnet_image, controlnet_inpaint_mask, tiler_kwargs):
        if isinstance(controlnet_image, Image.Image):
            controlnet_image = [controlnet_image] * len(self.controlnet.processors)

        controlnet_frames = []
        for i in range(len(self.controlnet.processors)):
            # image annotator
            image = self.controlnet.process_image(controlnet_image[i], processor_id=i)[0]
            if controlnet_inpaint_mask is not None and self.controlnet.processors[i].processor_id == "inpaint":
                image = self.apply_controlnet_mask_on_image(image, controlnet_inpaint_mask)

            # image to tensor
            image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)

            # vae encoder
            image = self.encode_image(image, **tiler_kwargs)
            if controlnet_inpaint_mask is not None and self.controlnet.processors[i].processor_id == "inpaint":
                image = self.apply_controlnet_mask_on_latents(image, controlnet_inpaint_mask)
            
            # store it
            controlnet_frames.append(image)
        return controlnet_frames


    def prepare_ipadapter_inputs(self, images, height=384, width=384):
        images = [image.convert("RGB").resize((width, height), resample=3) for image in images]
        images = [self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype) for image in images]
        return torch.cat(images, dim=0)


    def inpaint_fusion(self, latents, inpaint_latents, pred_noise, fg_mask, bg_mask, progress_id, background_weight=0.):
        # inpaint noise
        inpaint_noise = (latents - inpaint_latents) / self.scheduler.sigmas[progress_id]
        # merge noise
        weight = torch.ones_like(inpaint_noise)
        inpaint_noise[fg_mask] = pred_noise[fg_mask]
        inpaint_noise[bg_mask] += pred_noise[bg_mask] * background_weight
        weight[bg_mask] += background_weight
        inpaint_noise /= weight
        return inpaint_noise


    def preprocess_masks(self, masks, height, width, dim):
        out_masks = []
        for mask in masks:
            mask = self.preprocess_image(mask.resize((width, height), resample=Image.NEAREST)).mean(dim=1, keepdim=True) > 0
            mask = mask.repeat(1, dim, 1, 1).to(device=self.device, dtype=self.torch_dtype)
            out_masks.append(mask)
        return out_masks


    def prepare_entity_inputs(self, entity_prompts, entity_masks, width, height, t5_sequence_length=512, enable_eligen_inpaint=False):
        fg_mask, bg_mask = None, None
        if enable_eligen_inpaint:
            masks_ = deepcopy(entity_masks)
            fg_masks = torch.cat([self.preprocess_image(mask.resize((width//8, height//8))).mean(dim=1, keepdim=True) for mask in masks_])
            fg_masks = (fg_masks > 0).float()
            fg_mask = fg_masks.sum(dim=0, keepdim=True).repeat(1, 16, 1, 1) > 0
            bg_mask = ~fg_mask
        entity_masks = self.preprocess_masks(entity_masks, height//8, width//8, 1)
        entity_masks = torch.cat(entity_masks, dim=0).unsqueeze(0) # b, n_mask, c, h, w
        entity_prompts = self.encode_prompt(entity_prompts, t5_sequence_length=t5_sequence_length)['prompt_emb'].unsqueeze(0)
        return entity_prompts, entity_masks, fg_mask, bg_mask


    def prepare_latents(self, input_image, height, width, seed, tiled, tile_size, tile_stride):
        if input_image is not None:
            self.load_models_to_device(['vae_encoder'])
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.torch_dtype)
            input_latents = self.encode_image(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            noise = self.generate_noise((1, 16, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
            latents = self.scheduler.add_noise(input_latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = self.generate_noise((1, 16, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
            input_latents = None
        return latents, input_latents


    def prepare_ipadapter(self, ipadapter_images, ipadapter_scale):
        if ipadapter_images is not None:
            self.load_models_to_device(['ipadapter_image_encoder'])
            ipadapter_images = self.prepare_ipadapter_inputs(ipadapter_images)
            ipadapter_image_encoding = self.ipadapter_image_encoder(ipadapter_images).pooler_output
            self.load_models_to_device(['ipadapter'])
            ipadapter_kwargs_list_posi = {"ipadapter_kwargs_list": self.ipadapter(ipadapter_image_encoding, scale=ipadapter_scale)}
            ipadapter_kwargs_list_nega = {"ipadapter_kwargs_list": self.ipadapter(torch.zeros_like(ipadapter_image_encoding))}
        else:
            ipadapter_kwargs_list_posi, ipadapter_kwargs_list_nega = {"ipadapter_kwargs_list": {}}, {"ipadapter_kwargs_list": {}}
        return ipadapter_kwargs_list_posi, ipadapter_kwargs_list_nega


    def prepare_controlnet(self, controlnet_image, masks, controlnet_inpaint_mask, tiler_kwargs, enable_controlnet_on_negative):
        if controlnet_image is not None:
            self.load_models_to_device(['vae_encoder'])
            controlnet_kwargs_posi = {"controlnet_frames": self.prepare_controlnet_input(controlnet_image, controlnet_inpaint_mask, tiler_kwargs)}
            if len(masks) > 0 and controlnet_inpaint_mask is not None:
                print("The controlnet_inpaint_mask will be overridden by masks.")
                local_controlnet_kwargs = [{"controlnet_frames": self.prepare_controlnet_input(controlnet_image, mask, tiler_kwargs)} for mask in masks]
            else:
                local_controlnet_kwargs = None
        else:
            controlnet_kwargs_posi, local_controlnet_kwargs = {"controlnet_frames": None}, [{}] * len(masks)
        controlnet_kwargs_nega = controlnet_kwargs_posi if enable_controlnet_on_negative else {}
        return controlnet_kwargs_posi, controlnet_kwargs_nega, local_controlnet_kwargs


    def prepare_eligen(self, prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length, enable_eligen_inpaint, enable_eligen_on_negative, cfg_scale):
        if eligen_entity_masks is not None:
            entity_prompt_emb_posi, entity_masks_posi, fg_mask, bg_mask = self.prepare_entity_inputs(eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length, enable_eligen_inpaint)
            if enable_eligen_on_negative and cfg_scale != 1.0:
                entity_prompt_emb_nega = prompt_emb_nega['prompt_emb'].unsqueeze(1).repeat(1, entity_masks_posi.shape[1], 1, 1)
                entity_masks_nega = entity_masks_posi
            else:
                entity_prompt_emb_nega, entity_masks_nega = None, None
        else:
            entity_prompt_emb_posi, entity_masks_posi, entity_prompt_emb_nega, entity_masks_nega = None, None, None, None
            fg_mask, bg_mask = None, None
        eligen_kwargs_posi = {"entity_prompt_emb": entity_prompt_emb_posi, "entity_masks": entity_masks_posi}
        eligen_kwargs_nega = {"entity_prompt_emb": entity_prompt_emb_nega, "entity_masks": entity_masks_nega}
        return eligen_kwargs_posi, eligen_kwargs_nega, fg_mask, bg_mask


    def prepare_prompts(self, prompt, local_prompts, masks, mask_scales, t5_sequence_length, negative_prompt, cfg_scale):
        # Extend prompt
        self.load_models_to_device(['text_encoder_1', 'text_encoder_2'])
        prompt, local_prompts, masks, mask_scales = self.extend_prompt(prompt, local_prompts, masks, mask_scales)

        # Encode prompts
        prompt_emb_posi = self.encode_prompt(prompt, t5_sequence_length=t5_sequence_length)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False, t5_sequence_length=t5_sequence_length) if cfg_scale != 1.0 else None
        prompt_emb_locals = [self.encode_prompt(prompt_local, t5_sequence_length=t5_sequence_length) for prompt_local in local_prompts]
        return prompt_emb_posi, prompt_emb_nega, prompt_emb_locals
    
    
    def prepare_infinite_you(self, id_image, controlnet_image, infinityou_guidance, height, width):
        if self.infinityou_processor is not None and id_image is not None:
            return self.infinityou_processor.prepare_infinite_you(self.image_proj_model, id_image, controlnet_image, infinityou_guidance, height, width)
        else:
            return {}, controlnet_image


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt,
        negative_prompt="",
        cfg_scale=1.0,
        embedded_guidance=3.5,
        t5_sequence_length=512,
        # Image
        input_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
        seed=None,
        # Steps
        num_inference_steps=30,
        # local prompts
        local_prompts=(),
        masks=(),
        mask_scales=(),
        # ControlNet
        controlnet_image=None,
        controlnet_inpaint_mask=None,
        enable_controlnet_on_negative=False,
        # IP-Adapter
        ipadapter_images=None,
        ipadapter_scale=1.0,
        # EliGen
        eligen_entity_prompts=None,
        eligen_entity_masks=None,
        enable_eligen_on_negative=False,
        enable_eligen_inpaint=False,
        # InfiniteYou
        infinityou_id_image=None,
        infinityou_guidance=1.0,
        # TeaCache
        tea_cache_l1_thresh=None,
        # Tile
        tiled=False,
        tile_size=128,
        tile_stride=64,
        # Progress bar
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        height, width = self.check_resize_height_width(height, width)

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        latents, input_latents = self.prepare_latents(input_image, height, width, seed, tiled, tile_size, tile_stride)

        # Prompt
        prompt_emb_posi, prompt_emb_nega, prompt_emb_locals = self.prepare_prompts(prompt, local_prompts, masks, mask_scales, t5_sequence_length, negative_prompt, cfg_scale)

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # InfiniteYou
        infiniteyou_kwargs, controlnet_image = self.prepare_infinite_you(infinityou_id_image, controlnet_image, infinityou_guidance, height, width)
        
        # Entity control
        eligen_kwargs_posi, eligen_kwargs_nega, fg_mask, bg_mask = self.prepare_eligen(prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length, enable_eligen_inpaint, enable_eligen_on_negative, cfg_scale)

        # IP-Adapter
        ipadapter_kwargs_list_posi, ipadapter_kwargs_list_nega = self.prepare_ipadapter(ipadapter_images, ipadapter_scale)

        # ControlNets
        controlnet_kwargs_posi, controlnet_kwargs_nega, local_controlnet_kwargs = self.prepare_controlnet(controlnet_image, masks, controlnet_inpaint_mask, tiler_kwargs, enable_controlnet_on_negative)

        # TeaCache
        tea_cache_kwargs = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None}

        # Denoise
        self.load_models_to_device(['dit', 'controlnet'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Positive side
            inference_callback = lambda prompt_emb_posi, controlnet_kwargs: lets_dance_flux(
                dit=self.dit, controlnet=self.controlnet,
                hidden_states=latents, timestep=timestep,
                **prompt_emb_posi, **tiler_kwargs, **extra_input, **controlnet_kwargs, **ipadapter_kwargs_list_posi, **eligen_kwargs_posi, **tea_cache_kwargs, **infiniteyou_kwargs
            )
            noise_pred_posi = self.control_noise_via_local_prompts(
                prompt_emb_posi, prompt_emb_locals, masks, mask_scales, inference_callback,
                special_kwargs=controlnet_kwargs_posi, special_local_kwargs_list=local_controlnet_kwargs
            )

            # Inpaint
            if enable_eligen_inpaint:
                noise_pred_posi = self.inpaint_fusion(latents, input_latents, noise_pred_posi, fg_mask, bg_mask, progress_id)
            
            # Classifier-free guidance
            if cfg_scale != 1.0:
                # Negative side
                noise_pred_nega = lets_dance_flux(
                    dit=self.dit, controlnet=self.controlnet,
                    hidden_states=latents, timestep=timestep,
                    **prompt_emb_nega, **tiler_kwargs, **extra_input, **controlnet_kwargs_nega, **ipadapter_kwargs_list_nega, **eligen_kwargs_nega, **infiniteyou_kwargs,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Iterate
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        self.load_models_to_device(['vae_decoder'])
        image = self.decode_image(latents, **tiler_kwargs)

        # Offload all models
        self.load_models_to_device([])
        return image
    
    
    
class InfinitYou:
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        from facexlib.recognition import init_recognition_model
        from insightface.app import FaceAnalysis
        self.device = device
        self.torch_dtype = torch_dtype
        insightface_root_path = 'models/InfiniteYou/insightface'
        self.app_640 = FaceAnalysis(name='antelopev2', root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_640.prepare(ctx_id=0, det_size=(640, 640))
        self.app_320 = FaceAnalysis(name='antelopev2', root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_320.prepare(ctx_id=0, det_size=(320, 320))
        self.app_160 = FaceAnalysis(name='antelopev2', root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_160.prepare(ctx_id=0, det_size=(160, 160))
        self.arcface_model = init_recognition_model('arcface', device=self.device)
        
    def _detect_face(self, id_image_cv2):
        face_info = self.app_640.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
        face_info = self.app_320.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
        face_info = self.app_160.get(id_image_cv2)
        return face_info
    
    def extract_arcface_bgr_embedding(self, in_image, landmark):
        from insightface.utils import face_align
        arc_face_image = face_align.norm_crop(in_image, landmark=np.array(landmark), image_size=112)
        arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0, 3, 1, 2) / 255.
        arc_face_image = 2 * arc_face_image - 1
        arc_face_image = arc_face_image.contiguous().to(self.device)
        face_emb = self.arcface_model(arc_face_image)[0] # [512], normalized
        return face_emb
    
    def prepare_infinite_you(self, model, id_image, controlnet_image, infinityou_guidance, height, width):
        import cv2
        if id_image is None:
            return {'id_emb': None}, controlnet_image
        id_image_cv2 = cv2.cvtColor(np.array(id_image), cv2.COLOR_RGB2BGR)
        face_info = self._detect_face(id_image_cv2)
        if len(face_info) == 0:
            raise ValueError('No face detected in the input ID image')
        landmark = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]['kps'] # only use the maximum face
        id_emb = self.extract_arcface_bgr_embedding(id_image_cv2, landmark)
        id_emb = model(id_emb.unsqueeze(0).reshape([1, -1, 512]).to(dtype=self.torch_dtype))
        if controlnet_image is None:
            controlnet_image = Image.fromarray(np.zeros([height, width, 3]).astype(np.uint8))
        infinityou_guidance = torch.Tensor([infinityou_guidance]).to(device=self.device, dtype=self.torch_dtype)
        return {'id_emb': id_emb, 'infinityou_guidance': infinityou_guidance}, controlnet_image


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

    def check(self, dit: FluxDiT, hidden_states, conditioning):
        inp = hidden_states.clone()
        temb_ = conditioning.clone()
        modulated_inp, _, _, _, _ = dit.blocks[0].norm1_a(inp, emb=temb_)
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else: 
            coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
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
            self.previous_hidden_states = hidden_states.clone()
        return not should_calc
    
    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


def lets_dance_flux(
    dit: FluxDiT,
    controlnet: FluxMultiControlNetManager = None,
    hidden_states=None,
    timestep=None,
    prompt_emb=None,
    pooled_prompt_emb=None,
    guidance=None,
    text_ids=None,
    image_ids=None,
    controlnet_frames=None,
    tiled=False,
    tile_size=128,
    tile_stride=64,
    entity_prompt_emb=None,
    entity_masks=None,
    ipadapter_kwargs_list={},
    id_emb=None,
    infinityou_guidance=None,
    tea_cache: TeaCache = None,
    **kwargs
):
    if tiled:
        def flux_forward_fn(hl, hr, wl, wr):
            tiled_controlnet_frames = [f[:, :, hl: hr, wl: wr] for f in controlnet_frames] if controlnet_frames is not None else None
            return lets_dance_flux(
                dit=dit,
                controlnet=controlnet,
                hidden_states=hidden_states[:, :, hl: hr, wl: wr],
                timestep=timestep,
                prompt_emb=prompt_emb,
                pooled_prompt_emb=pooled_prompt_emb,
                guidance=guidance,
                text_ids=text_ids,
                image_ids=None,
                controlnet_frames=tiled_controlnet_frames,
                tiled=False,
                **kwargs
            )
        return FastTileWorker().tiled_forward(
            flux_forward_fn,
            hidden_states,
            tile_size=tile_size,
            tile_stride=tile_stride,
            tile_device=hidden_states.device,
            tile_dtype=hidden_states.dtype
        )


    # ControlNet
    if controlnet is not None and controlnet_frames is not None:
        controlnet_extra_kwargs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "prompt_emb": prompt_emb,
            "pooled_prompt_emb": pooled_prompt_emb,
            "guidance": guidance,
            "text_ids": text_ids,
            "image_ids": image_ids,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        if id_emb is not None:
            controlnet_text_ids = torch.zeros(id_emb.shape[0], id_emb.shape[1], 3).to(device=hidden_states.device, dtype=hidden_states.dtype)
            controlnet_extra_kwargs.update({"prompt_emb": id_emb, 'text_ids': controlnet_text_ids, 'guidance': infinityou_guidance})
        controlnet_res_stack, controlnet_single_res_stack = controlnet(
            controlnet_frames, **controlnet_extra_kwargs
        )

    if image_ids is None:
        image_ids = dit.prepare_image_ids(hidden_states)
    
    conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)
    if dit.guidance_embedder is not None:
        guidance = guidance * 1000
        conditioning = conditioning + dit.guidance_embedder(guidance, hidden_states.dtype)

    height, width = hidden_states.shape[-2:]
    hidden_states = dit.patchify(hidden_states)
    hidden_states = dit.x_embedder(hidden_states)

    if entity_prompt_emb is not None and entity_masks is not None:
        prompt_emb, image_rotary_emb, attention_mask = dit.process_entity_masks(hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids)
    else:
        prompt_emb = dit.context_embedder(prompt_emb)
        image_rotary_emb = dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
        attention_mask = None

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, hidden_states, conditioning)
    else:
        tea_cache_update = False

    if tea_cache_update:
        hidden_states = tea_cache.update(hidden_states)
    else:
        # Joint Blocks
        for block_id, block in enumerate(dit.blocks):
            hidden_states, prompt_emb = block(
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id, None)
            )
            # ControlNet
            if controlnet is not None and controlnet_frames is not None:
                hidden_states = hidden_states + controlnet_res_stack[block_id]

        # Single Blocks
        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        num_joint_blocks = len(dit.blocks)
        for block_id, block in enumerate(dit.single_blocks):
            hidden_states, prompt_emb = block(
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id + num_joint_blocks, None)
            )
            # ControlNet
            if controlnet is not None and controlnet_frames is not None:
                hidden_states[:, prompt_emb.shape[1]:] = hidden_states[:, prompt_emb.shape[1]:] + controlnet_single_res_stack[block_id]
        hidden_states = hidden_states[:, prompt_emb.shape[1]:]

        if tea_cache is not None:
            tea_cache.store(hidden_states)

    hidden_states = dit.final_norm_out(hidden_states, conditioning)
    hidden_states = dit.final_proj_out(hidden_states)
    hidden_states = dit.unpatchify(hidden_states, height, width)

    return hidden_states
