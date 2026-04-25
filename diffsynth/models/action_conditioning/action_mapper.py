import torch
from abc import ABC, abstractmethod


class ActionMapper(ABC):
    """
    Preprocessing step that converts raw action sequences into masked image
    sequences (trajectory visualisations) *before* they enter the DiT.

    This runs entirely outside the model graph.  Its output is VAE-encoded
    by the pipeline and then passed to ActionConditionedDiT as
    ``masked_latents``.

    Implement ``map()`` with your own rendering code, e.g.:
      - project 3-D end-effector poses onto the image plane
      - draw trajectory lines / gripper overlays on blank frames
      - render any other action-derived visual representation
    """

    @abstractmethod
    def map(self, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        actions : (B, T, action_dim)  raw action poses or delta actions

        Returns
        -------
        images : (B, C, T, H, W)  trajectory visualisation images in [-1, 1]
        """


class IdentityActionMapper(ActionMapper):
    """
    Placeholder that returns zero images.
    Replace with your trajectory rendering implementation.
    """

    def __init__(self, image_channels: int = 3, height: int = 256, width: int = 384):
        self.C = image_channels
        self.H = height
        self.W = width

    def map(self, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        B, T, _ = actions.shape
        return torch.zeros(
            B, self.C, T, self.H, self.W,
            device=actions.device,
            dtype=actions.dtype,
        )
