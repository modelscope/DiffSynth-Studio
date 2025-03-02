import numpy as np
import torch
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPImageProcessor
from transformers import CLIPConfig
from dataclasses import dataclass
from transformers import CLIPModel as HFCLIPModel
from safetensors.torch import load_file
from torch import nn, einsum

from .trainer.models.base_model import BaseModelConfig

from transformers import CLIPConfig
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from typing import Any, Optional, Tuple, Union, List
import torch

from .trainer.models.cross_modeling import Cross_model
from .trainer.models import clip_model
import torch.nn.functional as F
import gc
import json
from .config import MODEL_PATHS

class MPScore(torch.nn.Module):
    def __init__(self, device: Union[str, torch.device], path: str = MODEL_PATHS, condition: str = 'overall'):
        super().__init__()
        """Initialize the MPSModel with a processor, tokenizer, and model.

        Args:
            device (Union[str, torch.device]): The device to load the model on.
        """
        self.device = device
        processor_name_or_path = path.get("clip")
        self.image_processor = CLIPImageProcessor.from_pretrained(processor_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)
        self.model = clip_model.CLIPModel(processor_name_or_path, config_file=True)
        state_dict = load_file(path.get("mps"))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.condition = condition

    def _calculate_score(self, image: torch.Tensor, prompt: str) -> float:
        """Calculate the reward score for a single image and prompt.

        Args:
            image (torch.Tensor): The processed image tensor.
            prompt (str): The prompt text.

        Returns:
            float: The reward score.
        """
        def _tokenize(caption):
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            return input_ids

        text_input = _tokenize(prompt).to(self.device)
        if self.condition == 'overall':
            condition_prompt = 'light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things'
        elif self.condition == 'aesthetics':
            condition_prompt = 'light, color, clarity, tone, style, ambiance, artistry'
        elif self.condition == 'quality':
            condition_prompt = 'shape, face, hair, hands, limbs, structure, instance, texture'
        elif self.condition == 'semantic':
            condition_prompt = 'quantity, attributes, position, number, location'
        else:
            raise ValueError(
                f"Unsupported condition: {self.condition}. Choose 'overall', 'aesthetics', 'quality', or 'semantic'.")
        condition_batch = _tokenize(condition_prompt).repeat(text_input.shape[0], 1).to(self.device)

        with torch.no_grad():
            text_f, text_features = self.model.model.get_text_features(text_input)

            image_f = self.model.model.get_image_features(image.half())
            condition_f, _ = self.model.model.get_text_features(condition_batch)

            sim_text_condition = einsum('b i d, b j d -> b j i', text_f, condition_f)
            sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
            sim_text_condition = sim_text_condition / sim_text_condition.max()
            mask = torch.where(sim_text_condition > 0.3, 0, float('-inf'))
            mask = mask.repeat(1, image_f.shape[1], 1)
            image_features = self.model.cross_model(image_f, text_f, mask.half())[:, 0, :]

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_score = self.model.logit_scale.exp() * text_features @ image_features.T

        return image_score[0].cpu().numpy().item()

    @torch.no_grad()
    def score(self, images: Union[str, List[str], Image.Image, List[Image.Image]], prompt: str) -> List[float]:
        """Score the images based on the prompt.

        Args:
            images (Union[str, List[str], Image.Image, List[Image.Image]]): Path(s) to the image(s) or PIL image(s).
            prompt (str): The prompt text.

        Returns:
            List[float]: List of reward scores for the images.
        """
        if isinstance(images, (str, Image.Image)):
            # Single image
            if isinstance(images, str):
                image = self.image_processor(Image.open(images), return_tensors="pt")["pixel_values"].to(self.device)
            else:
                image = self.image_processor(images, return_tensors="pt")["pixel_values"].to(self.device)
            return [self._calculate_score(image, prompt)]
        elif isinstance(images, list):
            # Multiple images
            scores = []
            for one_images in images:
                if isinstance(one_images, str):
                    image = self.image_processor(Image.open(one_images), return_tensors="pt")["pixel_values"].to(self.device)
                elif isinstance(one_images, Image.Image):
                    image = self.image_processor(one_images, return_tensors="pt")["pixel_values"].to(self.device)
                else:
                    raise TypeError("The type of parameter images is illegal.")
                scores.append(self._calculate_score(image, prompt))
            return scores
        else:
            raise TypeError("The type of parameter images is illegal.")
