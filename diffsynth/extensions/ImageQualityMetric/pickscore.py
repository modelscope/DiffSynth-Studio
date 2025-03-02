import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import List, Union
import os
from .config import MODEL_PATHS

class PickScore(torch.nn.Module):
    def __init__(self, device: Union[str, torch.device], path: str = MODEL_PATHS):
        super().__init__()
        """Initialize the Selector with a processor and model.

        Args:
            device (Union[str, torch.device]): The device to load the model on.
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        processor_name_or_path = path.get("clip")
        model_pretrained_name_or_path = path.get("pickscore")
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(self.device)

    def _calculate_score(self, image: torch.Tensor, prompt: str, softmax: bool = False) -> float:
        """Calculate the score for a single image and prompt.

        Args:
            image (torch.Tensor): The processed image tensor.
            prompt (str): The prompt text.
            softmax (bool): Whether to apply softmax to the scores.

        Returns:
            float: The score for the image.
        """
        with torch.no_grad():
            # Prepare text inputs
            text_inputs = self.processor(
                text=prompt,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            # Embed images and text
            image_embs = self.model.get_image_features(pixel_values=image)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # Compute score
            score = (text_embs @ image_embs.T)[0]
            if softmax:
                # Apply logit scale and softmax
                score = torch.softmax(self.model.logit_scale.exp() * score, dim=-1)

        return score.cpu().item()

    @torch.no_grad()
    def score(self, images: Union[str, List[str], Image.Image, List[Image.Image]], prompt: str, softmax: bool = False) -> List[float]:
        """Score the images based on the prompt.

        Args:
            images (Union[str, List[str], Image.Image, List[Image.Image]]): Path(s) to the image(s) or PIL image(s).
            prompt (str): The prompt text.
            softmax (bool): Whether to apply softmax to the scores.

        Returns:
            List[float]: List of scores for the images.
        """
        try:
            if isinstance(images, (str, Image.Image)):
                # Single image
                if isinstance(images, str):
                    pil_image = Image.open(images)
                else:
                    pil_image = images

                # Prepare image inputs
                image_inputs = self.processor(
                    images=pil_image,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)

                return [self._calculate_score(image_inputs["pixel_values"], prompt, softmax)]
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

                    # Prepare image inputs
                    image_inputs = self.processor(
                        images=pil_image,
                        padding=True,
                        truncation=True,
                        max_length=77,
                        return_tensors="pt",
                    ).to(self.device)

                    scores.append(self._calculate_score(image_inputs["pixel_values"], prompt, softmax))
                return scores
            else:
                raise TypeError("The type of parameter images is illegal.")
        except Exception as e:
            raise RuntimeError(f"Error in scoring images: {e}")
