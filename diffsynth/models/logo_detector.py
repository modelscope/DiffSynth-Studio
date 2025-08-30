import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import GaussianBlur

class LogoDetector(nn.Module):
    def __init__(self, model_path=None, device="cuda"):
        super().__init__()
        self.device = device
        if model_path:
            # Load a custom model if a path is provided
            # For now, we assume the model is a Faster R-CNN model
            self.model = torch.load(model_path, map_location=self.device)
        else:
            # Load a pretrained Faster R-CNN model from torchvision
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image):
        # The image is expected to be a tensor of shape [C, H, W] with values in [0, 1]
        image = image.to(self.device)
        return image

    def get_logo_mask(self, image, threshold=0.5, blur_kernel_size=21, blur_sigma=5):
        # The image is expected to be a tensor of shape (B, C, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image_processed = self.preprocess_image(image)

        with torch.no_grad():
            predictions = self.model(image_processed)

        mask = torch.zeros(image.shape[2:], device=self.device)

        for i in range(len(predictions)):
            boxes = predictions[i]['boxes']
            scores = predictions[i]['scores']

            for box, score in zip(boxes, scores):
                if score > threshold:
                    x1, y1, x2, y2 = box.int()
                    mask[y1:y2, x1:x2] = 1

        # Create a soft mask by applying a Gaussian blur
        if blur_kernel_size > 0 and blur_sigma > 0:
            mask = F.gaussian_blur(mask.unsqueeze(0).unsqueeze(0), kernel_size=blur_kernel_size, sigma=blur_sigma)
            mask = mask.squeeze(0).squeeze(0)

        # Normalize the mask to be in the range [0, 1]
        if mask.max() > 0:
            mask = mask / mask.max()

        return mask
