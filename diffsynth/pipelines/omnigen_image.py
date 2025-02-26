from ..models.omnigen import OmniGenTransformer
from ..models.sdxl_vae_encoder import SDXLVAEEncoder
from ..models.sdxl_vae_decoder import SDXLVAEDecoder
from ..models.model_manager import ModelManager
from ..prompters.omnigen_prompter import OmniGenPrompter
from ..schedulers import FlowMatchScheduler
from .base import BasePipeline
from typing import Optional, Dict, Any, Tuple, List
from transformers.cache_utils import DynamicCache
import torch, os
from tqdm import tqdm



class OmniGenCache(DynamicCache):
    def __init__(self, 
                    num_tokens_for_img: int, offload_kv_cache: bool=False) -> None:
        if not torch.cuda.is_available():
            print("No available GPU, offload_kv_cache will be set to False, which will result in large memory usage and time cost when input multiple images!!!")
            offload_kv_cache = False
            raise RuntimeError("OffloadedCache can only be used with a GPU")
        super().__init__()
        self.original_device = []
        self.prefetch_stream = torch.cuda.Stream()
        self.num_tokens_for_img = num_tokens_for_img
        self.offload_kv_cache = offload_kv_cache

    def prefetch_layer(self, layer_idx: int):
        "Starts prefetching the next layer cache"
        if layer_idx < len(self):
            with torch.cuda.stream(self.prefetch_stream):
                # Prefetch next layer tensors to GPU
                device = self.original_device[layer_idx]
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device, non_blocking=True)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device, non_blocking=True)

    
    def evict_previous_layer(self, layer_idx: int):
        "Moves the previous layer cache to the CPU"
        if len(self) > 2:
            # We do it on the default stream so it occurs after all earlier computations on these tensors are done
            if layer_idx == 0: 
                prev_layer_idx = -1
            else:
                prev_layer_idx = (layer_idx - 1) % len(self)
            self.key_cache[prev_layer_idx] = self.key_cache[prev_layer_idx].to("cpu", non_blocking=True)
            self.value_cache[prev_layer_idx] = self.value_cache[prev_layer_idx].to("cpu", non_blocking=True)


    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        "Gets the cache for this layer to the device. Prefetches the next and evicts the previous layer."
        if layer_idx < len(self):
            if self.offload_kv_cache:
                # Evict the previous layer if necessary
                torch.cuda.current_stream().synchronize()
                self.evict_previous_layer(layer_idx)
                # Load current layer cache to its original device if not already there
                original_device = self.original_device[layer_idx]
                # self.prefetch_stream.synchronize(original_device)
                torch.cuda.synchronize(self.prefetch_stream)
                key_tensor = self.key_cache[layer_idx]
                value_tensor = self.value_cache[layer_idx]
                
                # Prefetch the next layer
                self.prefetch_layer((layer_idx + 1) % len(self))
            else:
                key_tensor = self.key_cache[layer_idx]
                value_tensor = self.value_cache[layer_idx]
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        
       
    def update(
        self,
        key_states: torch.Tensor, 
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if len(self.key_cache) < layer_idx:
            raise ValueError("OffloadedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            # only cache the states for condition tokens
            key_states = key_states[..., :-(self.num_tokens_for_img+1), :]
            value_states = value_states[..., :-(self.num_tokens_for_img+1), :]

             # Update the number of seen tokens
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]
                
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            if self.offload_kv_cache:
                self.evict_previous_layer(layer_idx)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            # only cache the states for condition tokens
            key_tensor, value_tensor = self[layer_idx]
            k = torch.cat([key_tensor, key_states], dim=-2)
            v = torch.cat([value_tensor, value_states], dim=-2)
            return k, v



class OmnigenImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(num_train_timesteps=1, shift=1, inverse_timesteps=True, sigma_min=0, sigma_max=1)
        # models
        self.vae_decoder: SDXLVAEDecoder = None
        self.vae_encoder: SDXLVAEEncoder = None
        self.transformer: OmniGenTransformer = None
        self.prompter: OmniGenPrompter = None
        self.model_names = ['transformer', 'vae_decoder', 'vae_encoder']


    def denoising_model(self):
        return self.transformer


    def fetch_models(self, model_manager: ModelManager, prompt_refiner_classes=[]):
        # Main models
        self.transformer, model_path = model_manager.fetch_model("omnigen_transformer", require_model_path=True)
        self.vae_decoder = model_manager.fetch_model("sdxl_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("sdxl_vae_encoder")
        self.prompter = OmniGenPrompter.from_pretrained(os.path.dirname(model_path))


    @staticmethod
    def from_model_manager(model_manager: ModelManager, prompt_refiner_classes=[], device=None):
        pipe = OmnigenImagePipeline(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_models(model_manager, prompt_refiner_classes=[])
        return pipe
    

    def encode_image(self, image, tiled=False, tile_size=64, tile_stride=32):
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    
    def encode_images(self, images, tiled=False, tile_size=64, tile_stride=32):
        latents = [self.encode_image(image.to(device=self.device), tiled, tile_size, tile_stride).to(self.torch_dtype) for image in images]
        return latents
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        return image
    

    def encode_prompt(self, prompt, clip_skip=1, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, clip_skip=clip_skip, device=self.device, positive=positive)
        return {"encoder_hidden_states": prompt_emb}
    

    def prepare_extra_input(self, latents=None):
        return {}
    

    def crop_position_ids_for_cache(self, position_ids, num_tokens_for_img):
        if isinstance(position_ids, list):
            for i in range(len(position_ids)):
                position_ids[i] = position_ids[i][:, -(num_tokens_for_img+1):]
        else:
            position_ids = position_ids[:, -(num_tokens_for_img+1):]
        return position_ids
    
    
    def crop_attention_mask_for_cache(self, attention_mask, num_tokens_for_img):
        if isinstance(attention_mask, list):
            return [x[..., -(num_tokens_for_img+1):, :] for x in attention_mask]
        return attention_mask[..., -(num_tokens_for_img+1):, :]
    

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        reference_images=[],
        cfg_scale=2.0,
        image_cfg_scale=2.0,
        use_kv_cache=True,
        offload_kv_cache=True,
        input_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
        num_inference_steps=20,
        tiled=False,
        tile_size=64,
        tile_stride=32,
        seed=None,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        height, width = self.check_resize_height_width(height, width)
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if input_image is not None:
            self.load_models_to_device(['vae_encoder'])
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.torch_dtype)
            latents = self.encode_image(image, **tiler_kwargs)
            noise = self.generate_noise((1, 4, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = self.generate_noise((1, 4, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        latents = latents.repeat(3, 1, 1, 1)

        # Encode prompts
        input_data = self.prompter(prompt, reference_images, height=height, width=width, use_img_cfg=True, separate_cfg_input=True, use_input_image_size_as_output=False)

        # Encode images
        reference_latents = [self.encode_images(images, **tiler_kwargs) for images in input_data['input_pixel_values']]
        
        # Pack all parameters
        model_kwargs = dict(input_ids=[input_ids.to(self.device) for input_ids in input_data['input_ids']], 
            input_img_latents=reference_latents, 
            input_image_sizes=input_data['input_image_sizes'], 
            attention_mask=[attention_mask.to(self.device) for attention_mask in input_data["attention_mask"]], 
            position_ids=[position_ids.to(self.device) for position_ids in input_data["position_ids"]], 
            cfg_scale=cfg_scale,
            img_cfg_scale=image_cfg_scale,
            use_img_cfg=True,
            use_kv_cache=use_kv_cache,
            offload_model=False,
        )
        
        # Denoise
        self.load_models_to_device(['transformer'])
        cache = [OmniGenCache(latents.size(-1)*latents.size(-2) // 4, offload_kv_cache) for _ in range(len(model_kwargs['input_ids']))] if use_kv_cache else None
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).repeat(latents.shape[0]).to(self.device)

            # Forward
            noise_pred, cache = self.transformer.forward_with_separate_cfg(latents, timestep, past_key_values=cache, **model_kwargs)

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

            # Update KV cache
            if progress_id == 0 and use_kv_cache:
                num_tokens_for_img = latents.size(-1)*latents.size(-2) // 4
                if isinstance(cache, list):
                    model_kwargs['input_ids'] = [None] * len(cache)
                else:
                    model_kwargs['input_ids'] = None
                model_kwargs['position_ids'] = self.crop_position_ids_for_cache(model_kwargs['position_ids'], num_tokens_for_img)
                model_kwargs['attention_mask'] = self.crop_attention_mask_for_cache(model_kwargs['attention_mask'], num_tokens_for_img)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        del cache
        self.load_models_to_device(['vae_decoder'])
        image = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        # offload all models
        self.load_models_to_device([])
        return image
