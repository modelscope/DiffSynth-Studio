from dataclasses import dataclass
from PIL import Image


@dataclass
class ControlNetInput:
    controlnet_id: int = 0
    scale: float = 1.0
    start: float = 1.0
    end: float = 0.0
    image: Image.Image = None
    inpaint_image: Image.Image = None
    inpaint_mask: Image.Image = None
    processor_id: str = None
