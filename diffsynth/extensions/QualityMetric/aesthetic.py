from typing import List, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
from safetensors.torch import load_file
import os
from typing import Union, List
from .config import MODEL_PATHS

class MLP(torch.nn.Module):
    def __init__(self, input_size: int, xcol: str = "emb", ycol: str = "avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 1024),
            #torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            #torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            #torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            #torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = torch.nn.functional.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = torch.nn.functional.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class AestheticScore:
    def __init__(self, device: torch.device, model_path: str = MODEL_PATHS.get("aesthetic_predictor")):
        """Initialize the Selector with a model and processor.

        Args:
            device (torch.device): The device to load the model on.
            model_path (str): Path to the model weights file.
        """
        self.device = device

        # Load the MLP model
        self.model = MLP(768)
        try:
            if model_path.endswith(".safetensors"):
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise ValueError(f"Error loading model weights from {model_path}: {e}")

        self.model.to(device)
        self.model.eval()

        # Load the CLIP model and processor
        clip_model_name = MODEL_PATHS.get('clip-large')
        self.model2 = AutoModel.from_pretrained(clip_model_name).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(clip_model_name)

    def _calculate_score(self, image: torch.Tensor) -> float:
        """Calculate the aesthetic score for a single image.

        Args:
            image (torch.Tensor): The processed image tensor.

        Returns:
            float: The aesthetic score.
        """
        with torch.no_grad():
            # Get image embeddings
            image_embs = self.model2.get_image_features(image)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            # Compute score
            score = self.model(image_embs).cpu().flatten().item()

        return score

    def score(self, images: Union[str, List[str], Image.Image, List[Image.Image]]) -> List[float]:
        """Score the images based on their aesthetic quality.

        Args:
            images (Union[str, List[str], Image.Image, List[Image.Image]]): Path(s) to the image(s) or PIL image(s).

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

                return [self._calculate_score(image_inputs["pixel_values"])]
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

                    scores.append(self._calculate_score(image_inputs["pixel_values"]))
                return scores
            else:
                raise TypeError("The type of parameter images is illegal.")
        except Exception as e:
            raise RuntimeError(f"Error in scoring images: {e}")
