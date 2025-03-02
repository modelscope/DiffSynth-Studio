from typing import List, Union
from PIL import Image
import torch
from .open_clip import create_model_and_transforms, get_tokenizer
from .config import MODEL_PATHS

class CLIPScore(torch.nn.Module):
    def __init__(self, device: torch.device, path: str = MODEL_PATHS):
        super().__init__()
        """Initialize the CLIPScore with a model and tokenizer.
        
        Args:
            device (torch.device): The device to load the model on.
        """
        self.device = device

        # Create model and transforms
        self.model, _, self.preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            # "laion2B-s32B-b79K",
            pretrained=path.get("open_clip"),
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )

        # Initialize tokenizer
        self.tokenizer = get_tokenizer("ViT-H-14", path["open_clip_bpe"])
        self.model = self.model.to(device)
        self.model.eval()

    def _calculate_score(self, image: torch.Tensor, prompt: str) -> float:
        """Calculate the CLIP score for a single image and prompt.

        Args:
            image (torch.Tensor): The processed image tensor.
            prompt (str): The prompt text.

        Returns:
            float: The CLIP score.
        """
        with torch.no_grad():
            # Process the prompt
            text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)

            # Calculate the CLIP score
            outputs = self.model(image, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            clip_score = torch.diagonal(logits_per_image).cpu().numpy()

        return clip_score[0].item()

    @torch.no_grad()
    def score(self, images: Union[str, List[str], Image.Image, List[Image.Image]], prompt: str) -> List[float]:
        """Score the images based on the prompt.

        Args:
            images (Union[str, List[str], Image.Image, List[Image.Image]]): Path(s) to the image(s) or PIL image(s).
            prompt (str): The prompt text.

        Returns:
            List[float]: List of CLIP scores for the images.
        """
        if isinstance(images, (str, Image.Image)):
            # Single image
            if isinstance(images, str):
                image = self.preprocess_val(Image.open(images)).unsqueeze(0).to(device=self.device, non_blocking=True)
            else:
                image = self.preprocess_val(images).unsqueeze(0).to(device=self.device, non_blocking=True)
            return [self._calculate_score(image, prompt)]
        elif isinstance(images, list):
            # Multiple images
            scores = []
            for one_images in images:
                if isinstance(one_images, str):
                    image = self.preprocess_val(Image.open(one_images)).unsqueeze(0).to(device=self.device, non_blocking=True)
                elif isinstance(one_images, Image.Image):
                    image = self.preprocess_val(one_images).unsqueeze(0).to(device=self.device, non_blocking=True)
                else:
                    raise TypeError("The type of parameter images is illegal.")
                scores.append(self._calculate_score(image, prompt))
            return scores
        else:
            raise TypeError("The type of parameter images is illegal.")
