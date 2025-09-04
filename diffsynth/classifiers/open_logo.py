import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_vgg16_bn
from torchvision.ops import nms

class OpenLogoClassifier(nn.Module):
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        self.device = device
        # Load the pretrained Faster R-CNN model with VGG16 backbone
        self.model = fasterrcnn_vgg16_bn(pretrained=False) # We will load our own weights

        # Load the state dict from the provided .pt file
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image):
        # Preprocess the image for the Faster R-CNN model
        # The image is expected to be a tensor of shape [C, H, W] with values in [0, 1]
        image = image.to(self.device)
        return image

    def compute_gradient(self, latents, pipeline, timestep, class_id, mask=None, guidance_scale=1.0):
        with torch.enable_grad():
            latents = latents.detach().requires_grad_(True)

            # Decode latents to video
            video = pipeline.vae.decode(latents)

            # The output of the VAE is in the range [-1, 1], we need to normalize it to [0, 1]
            video = (video / 2 + 0.5).clamp(0, 1)

            log_prob_sum = 0

            num_frames = video.shape[2]
            for i in range(num_frames):
                image = video[:, :, i, :, :]

                # Preprocess the image for the classifier
                image_processed = self.preprocess_image(image)

                # Get predictions from the classifier
                self.model.train()

                image_for_model = image_processed
                features = self.model.backbone(image_for_model)
                proposals, proposal_losses = self.model.rpn(image_for_model, features, None)
                box_features = self.model.roi_heads.box_roi_pool(features, proposals, [image_for_model.shape[2:]])
                box_features = self.model.roi_heads.box_head(box_features)
                class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)

                log_probs = torch.nn.functional.log_softmax(class_logits, dim=-1)
                log_prob_target = log_probs[:, class_id].mean()
                log_prob_sum = log_prob_sum + log_prob_target

            grad = torch.autograd.grad(log_prob_sum, latents, grad_outputs=torch.ones_like(log_prob_sum))[0]

            # Apply the soft logo mask if provided
            if mask is not None:
                grad = grad * mask

            # Clamp the gradient to avoid artifacts
            grad = torch.clamp(grad, -0.1, 0.1)

            self.model.eval()

        return grad
