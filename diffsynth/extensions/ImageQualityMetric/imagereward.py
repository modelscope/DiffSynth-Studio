import os
import torch
from PIL import Image
from typing import List, Union
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .BLIP.blip_pretrain import BLIP_Pretrain
from torchvision.transforms import InterpolationMode
from safetensors.torch import load_file
from .config import MODEL_PATHS
BICUBIC = InterpolationMode.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class MLP(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            #nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            #nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            #nn.ReLU(),
            torch.nn.Linear(16, 1)
        )
        
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return self.layers(input)

class ImageReward(torch.nn.Module):
    def __init__(self, med_config, device='cpu', bert_model_path=""):
        super().__init__()
        self.device = device
        
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config, bert_model_path=bert_model_path)
        self.preprocess = _transform(224)
        self.mlp = MLP(768)
        
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    def score_grad(self, prompt_ids, prompt_attention_mask, image):
        """Calculate the score with gradient for a single image and prompt.

        Args:
            prompt_ids (torch.Tensor): Tokenized prompt IDs.
            prompt_attention_mask (torch.Tensor): Attention mask for the prompt.
            image (torch.Tensor): The processed image tensor.

        Returns:
            torch.Tensor: The reward score.
        """
        image_embeds = self.blip.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(
            prompt_ids,
            attention_mask=prompt_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :]
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        return rewards

    def score(self, images: Union[str, List[str], Image.Image, List[Image.Image]], prompt: str = "") -> List[float]:
        """Score the images based on the prompt.

        Args:
            prompt (str): The prompt text.
            images (Union[str, List[str], Image.Image, List[Image.Image]]): Path(s) to the image(s) or PIL image(s).

        Returns:
            List[float]: List of scores for the images.
        """
        if isinstance(images, (str, Image.Image)):
            # Single image
            if isinstance(images, str):
                pil_image = Image.open(images)
            else:
                pil_image = images
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            return [self._calculate_score(prompt, image).item()]
        elif isinstance(images, list):
            # Multiple images
            scores = []
            for one_image in images:
                if isinstance(one_image, str):
                    pil_image = Image.open(one_image)
                elif isinstance(one_image, Image.Image):
                    pil_image = one_image
                else:
                    raise TypeError("The type of parameter images is illegal.")
                image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                scores.append(self._calculate_score(prompt, image).item())
            return scores
        else:
            raise TypeError("The type of parameter images is illegal.")

    def _calculate_score(self, prompt: str, image: torch.Tensor) -> torch.Tensor:
        """Calculate the score for a single image and prompt.

        Args:
            prompt (str): The prompt text.
            image (torch.Tensor): The processed image tensor.

        Returns:
            torch.Tensor: The reward score.
        """
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        image_embeds = self.blip.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].float()
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        return rewards

    def inference_rank(self, prompt: str, generations_list: List[Union[str, Image.Image]]) -> tuple:
        """Rank the images based on the prompt.

        Args:
            prompt (str): The prompt text.
            generations_list (List[Union[str, Image.Image]]): List of image paths or PIL images.

        Returns:
            tuple: (indices, rewards) where indices are the ranks and rewards are the scores.
        """
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        txt_set = []
        for generation in generations_list:
            if isinstance(generation, str):
                pil_image = Image.open(generation)
            elif isinstance(generation, Image.Image):
                pil_image = generation
            else:
                raise TypeError("The type of parameter generations_list is illegal.")
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            text_output = self.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_set.append(text_output.last_hidden_state[:, 0, :])
        txt_features = torch.cat(txt_set, 0).float()
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()


class ImageRewardScore(torch.nn.Module):
    def __init__(self, device: Union[str, torch.device], path: str = MODEL_PATHS):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        model_path = path.get("imagereward")
        med_config = path.get("med_config")
        state_dict = load_file(model_path)
        self.model = ImageReward(device=self.device, med_config=med_config, bert_model_path=path.get("bert_model_path")).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    @torch.no_grad()
    def score(self, images: Union[str, List[str], Image.Image, List[Image.Image]], prompt: str) -> List[float]:
        """Score the images based on the prompt.

        Args:
            images (Union[str, List[str], Image.Image, List[Image.Image]]): Path(s) to the image(s) or PIL image(s).
            prompt (str): The prompt text.

        Returns:
            List[float]: List of scores for the images.
        """
        return self.model.score(images, prompt)
