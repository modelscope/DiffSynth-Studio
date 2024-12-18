from ..models import ModelManager, SD3TextEncoder1, HunyuanVideoVAEDecoder
from .base import BasePipeline
from ..prompters import HunyuanVideoPrompter
import torch
from transformers import LlamaModel
from einops import rearrange
import numpy as np
from tqdm import tqdm
from PIL import Image


class HunyuanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        # 参照diffsynth的排序，text_encoder_1指CLIP；text_encoder_2指llm，与hunyuanvideo源代码刚好相反
        self.prompter = HunyuanVideoPrompter()
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: LlamaModel = None
        self.vae_decoder: HunyuanVideoVAEDecoder = None
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'vae_decoder']

    def fetch_models(self, model_manager: ModelManager):
        self.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("hunyuan_video_text_encoder_2")
        self.vae_decoder = model_manager.fetch_model("hunyuan_video_vae_decoder")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)

    @staticmethod
    def from_model_manager(model_manager: ModelManager, device=None):

        pipe = HunyuanVideoPipeline(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_models(model_manager)
        return pipe

    def encode_prompt(self, prompt, positive=True, clip_sequence_length=77, llm_sequence_length=256):
        prompt_emb, pooled_prompt_emb = self.prompter.encode_prompt(prompt,
                                                                    device=self.device,
                                                                    positive=positive,
                                                                    clip_sequence_length=clip_sequence_length,
                                                                    llm_sequence_length=llm_sequence_length)
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        seed=None,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # encode prompt
        # prompt_emb_posi = self.encode_prompt(prompt, positive=True)

        # test data
        latents = torch.load('latents.pt').to(device=self.device, dtype=self.torch_dtype)  # torch.Size([1, 16, 33, 90, 160])
        latents = latents[:, :, :2, :, :]
        # Tiler parameters
        tiler_kwargs = dict(use_temporal_tiling=False, use_spatial_tiling=False, sample_ssize=256, sample_tsize=64)
        # decode
        self.load_models_to_device(['vae_decoder'])
        frames = self.vae_decoder.decode_video(latents, **tiler_kwargs)
        frames = self.tensor2video(frames[0])
        return frames
